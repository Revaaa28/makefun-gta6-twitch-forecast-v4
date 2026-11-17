# makefun-gta6-twitch-forecast-v4

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Revaaa28/makefun-gta6-twitch-forecast-v4/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Revaaa28/makefun-gta6-twitch-forecast-v4/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A launch-day Twitch viewer forecaster for Grand Theft Auto VI. Given everything recorded up to the end of hour 1 of release day, it forecasts five quantiles (5%, 25%, 50%, 75%, 95%) of the Twitch category's viewer count for each of hours 2 to 24.

## How it works

1. **Pool selection by unlock type.** The case file carries dozens of past big game launches with their full hourly curves. The program keeps only the launches whose unlock was a *rolling midnight local* (console launches like GTA VI, no simultaneous global unlock, no target-history pseudo-launch): those share GTA VI's day shape, while simultaneous-global launches follow a different curve and poison a pooled shape. If the rolling pool ever came back too small, it widens to already-launched comparables, then to all complete ones.
2. **Local intercepts + one smooth size slope.** A joint least-squares fit of every pool curve's log ratio log(viewers_h / viewers_1) on a quadratic basis in the hour — the basis also interacted with (log10 size − target log10 size) — yields a single smooth slope curve: how the whole day shape tilts with launch size. Each hour's centre is that hour's pooled mean ratio plus the smooth slope times the target's size offset. Per-hour independent slopes proved noisy with eight pool curves, and size-kernel weighting threw away most of the pool; both measurably lost to this combination.
3. **Quantiles from pooled standardised residuals.** Each pool curve's residual against its fitted line is divided by its hour's residual spread (the spread curve smoothed over neighbouring hours — eight residuals per hour estimate it poorly), the standardised residuals are pooled within three hour blocks (early / mid / late day), and their interpolated quantiles, rescaled per hour and calibrated by a 0.95 spread factor, give the five forecast quantiles.

## Validation

Graded locally on the kit's four practice launches against the real answers, using the exact contest pinball formula (log scale, levels 0.05/0.25/0.5/0.75/0.95):

| model | p00 | p01 | p02 | p03 | mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| **this entry (v4)** | 0.1869 | 0.1380 | 0.1033 | 0.1973 | **0.1564** |
| previous entry (v3, kernel local-linear) | 0.2302 | 0.1538 | 0.1603 | 0.2457 | 0.1975 |
| regression on launched pool | 0.4646 | 0.2470 | 0.1370 | 0.1888 | 0.2593 |
| median shape on rolling pool | 0.4013 | 0.1540 | 0.2080 | 0.2352 | 0.2496 |
| house persistence baseline | 0.7437 | 0.2868 | 0.2146 | 0.3033 | 0.3871 |

Anti-overfit checks: leave-one-out over the eight pool curves gives 0.3444 (previous entry 0.3410), and the mimicked final — each practice launch scored with the other three added to its training pool, the shape of the real final case — gives 0.1562 against the previous entry's 0.1700. Known soft spot: the fourth practice launch's late-day surge still escapes the 90% band (coverage 0.70 there, 1.00 on the other three). Deterministic, Python standard library only, no network, ~0.1 s per case.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=revaaa28&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
