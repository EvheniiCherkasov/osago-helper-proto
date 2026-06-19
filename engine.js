/* engine.js — deterministic recommendation engine for the OSCPV offer list.
 *
 * Pure functions, no I/O, no LLM. Works in the browser (window.HelpMeChoose)
 * and under Node (module.exports) so tests/test_engine.js can require it.
 *
 * Public API:
 *   recommend(offers, answers) -> { intent, tolerance, filter, scores,
 *                                   primary, alternatives }
 *   answers = { intent: 'cheapest'|'best_reviews'|'fastest_payouts'|'balanced',
 *               price_tolerance: 'no'|'p10'|'p20'|null }
 *
 * Config: CONFIG.ALLOW_COMMERCIAL_BIAS (default false). When false,
 * `commercial_boost` provably never affects ranking. When true it is used
 * ONLY as the last tie-breaker, never added to any score.
 */
(function (global) {
  'use strict';

  var CONFIG = { ALLOW_COMMERCIAL_BIAS: false };

  var INTENT_WEIGHTS = {
    cheapest:        { price: 0.70, rating: 0.10, payout: 0.10, reviews: 0.10 },
    best_reviews:    { price: 0.10, rating: 0.20, payout: 0.10, reviews: 0.60 },
    fastest_payouts: { price: 0.10, rating: 0.20, payout: 0.60, reviews: 0.10 },
    balanced:        { price: 0.30, rating: 0.25, payout: 0.20, reviews: 0.25 }
  };

  // mirrors the AI-review-summary aspect keys (integration seam #1)
  var ASPECT_WEIGHTS = {
    'виплати': 0.35,
    'швидкість_врегулювання': 0.25,
    'підтримка': 0.15,
    'асистанс': 0.15,
    'оформлення': 0.10
  };

  var TOLERANCE_MULT = { no: 1.05, p10: 1.10, p20: 1.20 };
  // relaxation ladder: if a tier leaves <3 offers, step to the next
  var TIER_ORDER = ['no', 'p10', 'p20', 'none'];
  var TIER_LABEL = { no: '+5%', p10: '+10%', p20: '+20%', none: 'без обмеження' };

  function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

  function median(nums) {
    var s = nums.slice().sort(function (a, b) { return a - b; });
    var n = s.length, m = Math.floor(n / 2);
    return n % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  function aspectMean(aspects) {
    // Weighted mean over the aspects that are PRESENT. A missing aspect (e.g.
    // "асистанс" dropped when <10 mentions in the real review cache) is excluded
    // and the weights renormalize — it is not treated as a 0 score. With all
    // five aspects present (mock data) the denominator is 1.0, so this is a
    // no-op there.
    var sum = 0, wsum = 0;
    for (var k in ASPECT_WEIGHTS) {
      if (ASPECT_WEIGHTS.hasOwnProperty(k) && aspects[k] != null) {
        sum += aspects[k] * ASPECT_WEIGHTS[k];
        wsum += ASPECT_WEIGHTS[k];
      }
    }
    return wsum > 0 ? sum / wsum : 0;
  }

  function volumeFactor(count) {
    if (!count || count <= 1) return 0;
    return Math.min(1, Math.log(count) / Math.LN10 / 3); // log10(count)/3
  }

  /* Normalize every dimension to 0..1 across the given (already filtered)
   * offer list. Returns { dims: {id -> {price,rating,payout,reviews}}, ctx }. */
  function computeDims(offers) {
    var prices = offers.map(function (o) { return o.price_uah; });
    var minP = Math.min.apply(null, prices), maxP = Math.max.apply(null, prices);
    var pSpan = (maxP - minP) || 1;

    // Payout source. Official MTSBU avg_payout_days (mock dataset) takes
    // precedence — inverse-normalized + above-median complaint penalty. When
    // days are absent for the whole list (real data from the AI-review cache),
    // fall back to the reviews-based settlement_speed_score (0..1, higher better)
    // so we never invent a day count we don't actually have.
    var useDays = offers.every(function (o) { return o.mtsbu.avg_payout_days != null; });
    var minD = null, maxD = null, dSpan = 1, medC = 0, maxC = 0, cSpan = 1;
    if (useDays) {
      var days = offers.map(function (o) { return o.mtsbu.avg_payout_days; });
      var complaints = offers.map(function (o) { return o.mtsbu.complaints_per_10k; })
        .filter(function (c) { return c != null; });
      minD = Math.min.apply(null, days); maxD = Math.max.apply(null, days);
      dSpan = (maxD - minD) || 1;
      if (complaints.length) {
        maxC = Math.max.apply(null, complaints); medC = median(complaints);
        cSpan = (maxC - medC) || 1;
      }
    }

    var dims = {};
    offers.forEach(function (o) {
      var price = clamp((maxP - o.price_uah) / pSpan, 0, 1);
      var rating = clamp(o.mtsbu.rating / 5, 0, 1);

      var payout, penalty = 0;
      if (useDays) {
        // faster (fewer days) is better, then penalize above-median complaints by up to -0.15
        var payoutRaw = (maxD - o.mtsbu.avg_payout_days) / dSpan;
        if (o.mtsbu.complaints_per_10k != null && o.mtsbu.complaints_per_10k > medC) {
          penalty = clamp(0.15 * (o.mtsbu.complaints_per_10k - medC) / cSpan, 0, 0.15);
        }
        payout = clamp(payoutRaw - penalty, 0, 1);
      } else {
        payout = clamp(o.mtsbu.settlement_speed_score != null ? o.mtsbu.settlement_speed_score : 0, 0, 1);
      }

      var reviews = clamp(aspectMean(o.reviews.aspects) * volumeFactor(o.reviews.count), 0, 1);

      dims[o.id] = {
        price: price, rating: rating, payout: payout, reviews: reviews,
        complaintPenalty: penalty
      };
    });

    return {
      dims: dims,
      ctx: { minP: minP, maxP: maxP, minD: minD, maxD: maxD, medC: medC, useDays: useDays }
    };
  }

  /* Q2 price-tolerance hard filter, applied BEFORE scoring. Steps down the
   * relaxation ladder until >=3 offers remain (or the filter is removed). */
  function applyToleranceFilter(offers, tolerance) {
    var minP = Math.min.apply(null, offers.map(function (o) { return o.price_uah; }));
    var start = TIER_ORDER.indexOf(tolerance);
    if (start < 0) start = 0;

    for (var i = start; i < TIER_ORDER.length; i++) {
      var tier = TIER_ORDER[i];
      var subset;
      if (tier === 'none') {
        subset = offers.slice();
      } else {
        var cap = minP * TOLERANCE_MULT[tier];
        subset = offers.filter(function (o) { return o.price_uah <= cap + 1e-9; });
      }
      if (subset.length >= 3 || tier === 'none') {
        return {
          offers: subset,
          requested: tolerance,
          tier_used: tier,
          relaxed: i > start,
          minP: minP,
          count_before: offers.length,
          count_after: subset.length,
          relaxed_note: (i > start)
            ? (tier === 'none'
                ? 'Показали всі пропозиції — у межах вашого порогу було менше 3.'
                : 'Розширили діапазон до ' + TIER_LABEL[tier] +
                  ', бо в межах вашого порогу було менше 3 пропозицій.')
            : null
        };
      }
    }
    // unreachable in practice (full list >= 3)
    return {
      offers: offers.slice(), requested: tolerance, tier_used: 'none',
      relaxed: true, minP: minP, count_before: offers.length,
      count_after: offers.length, relaxed_note: null
    };
  }

  /* The 2-3 strongest dimensions for this intent + the raw numbers behind them.
   * Feeds the result chips ("виплати ~28 дн.", "+180 грн до найдешевшої") and
   * the offline explanation layer. */
  function reasonFacts(offer, dim, weights, minP) {
    var contrib = {
      price: weights.price * dim.price,
      rating: weights.rating * dim.rating,
      payout: weights.payout * dim.payout,
      reviews: weights.reviews * dim.reviews
    };
    var ordered = Object.keys(contrib)
      .filter(function (k) { return contrib[k] > 0; })
      .sort(function (a, b) { return contrib[b] - contrib[a]; });
    var top = ordered.slice(0, 3);

    var asp = offer.reviews.aspects;
    return {
      top_dims: top,
      contributions: contrib,
      facts: {
        price_uah: offer.price_uah,
        price_delta_uah: offer.price_uah - minP,
        rating: offer.mtsbu.rating,
        avg_payout_days: offer.mtsbu.avg_payout_days,
        settlement_speed_score: (offer.mtsbu.settlement_speed_score != null ? offer.mtsbu.settlement_speed_score : null),
        complaints_per_10k: offer.mtsbu.complaints_per_10k,
        review_count: offer.reviews.count,
        pos_payouts: asp['виплати'],
        pos_speed: asp['швидкість_врегулювання'],
        pos_support: asp['підтримка'],
        pos_assist: asp['асистанс']
      }
    };
  }

  /* Deterministic ranking comparator. score desc -> rating desc -> price asc;
   * commercial_boost only when ALLOW_COMMERCIAL_BIAS, strictly as last tie-break. */
  function makeComparator() {
    var bias = CONFIG.ALLOW_COMMERCIAL_BIAS;
    return function (a, b) {
      if (Math.abs(a.score - b.score) > 1e-9) return b.score - a.score;
      if (Math.abs(a.offer.mtsbu.rating - b.offer.mtsbu.rating) > 1e-9) {
        return b.offer.mtsbu.rating - a.offer.mtsbu.rating;
      }
      if (a.offer.price_uah !== b.offer.price_uah) {
        return a.offer.price_uah - b.offer.price_uah;
      }
      if (bias) {
        var ab = a.offer.commercial_boost ? 1 : 0;
        var bb = b.offer.commercial_boost ? 1 : 0;
        if (ab !== bb) return bb - ab;
      }
      return 0; // stable: preserve input order
    };
  }

  function recommend(offers, answers) {
    answers = answers || {};
    var intent = answers.intent || 'balanced';
    var weights = INTENT_WEIGHTS[intent] || INTENT_WEIGHTS.balanced;

    // Q2 is skipped for intent=cheapest (price already dominates)
    var tolerance = (intent === 'cheapest') ? null : (answers.price_tolerance || null);

    var filter;
    if (tolerance) {
      filter = applyToleranceFilter(offers, tolerance);
    } else {
      filter = {
        offers: offers.slice(), requested: null, tier_used: null,
        relaxed: false, minP: Math.min.apply(null, offers.map(function (o) { return o.price_uah; })),
        count_before: offers.length, count_after: offers.length, relaxed_note: null
      };
    }

    var pool = filter.offers;
    var nd = computeDims(pool);
    var minP = nd.ctx.minP;

    var scored = pool.map(function (o) {
      var d = nd.dims[o.id];
      var score = weights.price * d.price + weights.rating * d.rating +
                  weights.payout * d.payout + weights.reviews * d.reviews;
      return {
        id: o.id,
        insurer: o.insurer,
        score: score,
        offer: o,
        dims: { price: d.price, rating: d.rating, payout: d.payout, reviews: d.reviews },
        weights: weights,
        contributions: {
          price: weights.price * d.price,
          rating: weights.rating * d.rating,
          payout: weights.payout * d.payout,
          reviews: weights.reviews * d.reviews
        },
        reason_facts: reasonFacts(o, d, weights, minP)
      };
    });

    scored.sort(makeComparator());

    return {
      intent: intent,
      tolerance: tolerance,
      filter: filter,
      scores: scored,
      primary: scored[0] || null,
      alternatives: scored.slice(1, 3)
    };
  }

  var api = {
    recommend: recommend,
    computeDims: computeDims,
    applyToleranceFilter: applyToleranceFilter,
    CONFIG: CONFIG,
    INTENT_WEIGHTS: INTENT_WEIGHTS,
    ASPECT_WEIGHTS: ASPECT_WEIGHTS,
    TIER_LABEL: TIER_LABEL
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  } else {
    global.HelpMeChoose = api;
  }
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));
