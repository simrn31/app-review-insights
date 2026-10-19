# Management Summary - App Review Insights

*Data source: Google Play Store (com.spotify.music). Reviews analysed: 2,921 (2026-10-10 to 2026-10-14). Sentiment engine: VADER.*

## Headline
- Average rating: **4.04**; 20% of users are unsatisfied (1-2★), 74% satisfied (4-5★).
- Rating and text sentiment agree (Spearman ρ = **0.48**).

## Did something change?
- Not enough weekly data in this selection to detect a shift.

## Top complaint themes (1-2★ reviews)
- **Pricing & Subscription** - 29.3% of complaints
- **Ads** - 24.7% of complaints
- **Crashes & Bugs** - 6.1% of complaints
- **Performance & Battery** - 4.7% of complaints
- **UI / UX** - 1.4% of complaints

## Most frequent complaint phrases (bigrams)
- "play song" (29×)
- "many ad" (24×)
- "every song" (23×)
- "song not" (22×)
- "listen music" (20×)
- "free version" (20×)
- "ad every" (20×)
- "listen song" (19×)

## Words most associated with 1-2★ (vs 4-5★) reviews
ad, not, song, premium, play, pay, worst, cannot, many, even

## Recommended actions
1. Triage the fastest-growing theme first - it is most likely tied to the latest release.
2. Add automated tests/monitoring for the top bug-type phrases above.
3. Re-run this pipeline weekly to track the trend.
