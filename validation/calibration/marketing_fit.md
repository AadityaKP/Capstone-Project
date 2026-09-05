# F1 marketing-curve fit (CAL only)

- **Fitted `SATURATION_ACQUISITION_RATE_V2` = 0.0727** (was 0.20, ASSUMED)
- CAL median |4q growth error| at fit: **12.9 pp** (v1 assumed value's
  loss on the same grid: 52.5 pp at the nearest grid point)
- 95% bootstrap CI over CAL companies (2,000 resamples, argmin per resample):
  **[0.0475, 0.1113]**
- Protocol: hold arm, 10 matched seeds, financing_enabled=True (so the three
  CAL companies with un-modelled financing deaths contribute), split per
  `panel_split.csv`, HOLDOUT untouched.
- Loss curve: `f1_loss_curve.csv`; per-company signed errors:
  `f1_per_company_errors.csv`.

## Churn-band sensitivity at the fitted rate (reported, NOT fit - D2 showed
churn is a minor term; the band median is an assumption of the mapping)

 flat_churn  median_abs_error_pp  median_signed_error_pp
      0.020            13.834396              -10.952764
      0.027            22.319254              -20.684889
      0.034            29.935581              -29.935581

## Loss curve

 saturation_rate  median_abs_error_pp  mean_abs_error_pp  median_signed_error_pp  n_evaluable
         0.01000            32.139569          34.138528              -32.139569           20
         0.01152            31.460917          33.557916              -31.460917           20
         0.01328            30.673594          32.883524              -30.673594           20
         0.01530            29.770688          32.112638              -29.770688           20
         0.01763            28.729765          31.291999              -28.729765           20
         0.02031            27.532665          30.383448              -27.532665           20
         0.02340            26.151974          29.337499              -26.151974           20
         0.02697            24.556305          28.133546              -24.556305           20
         0.03107            23.498823          26.803340              -22.710797           20
         0.03580            21.549815          25.451183              -20.566666           20
         0.04125            19.301729          23.890688              -18.085601           20
         0.04754            16.701126          22.301364              -15.205675           20
         0.05477            14.360228          21.060117              -11.867827           20
         0.06311            13.986506          19.614156               -8.371380           20
         0.07272            12.914475          17.929607               -4.458168           20
         0.08379            15.738201          17.483486                0.115588           20
         0.09655            14.561726          18.743649                5.492998           20
         0.11125            15.865256          21.084600               11.805814           20
         0.12819            21.678331          24.800118               19.264948           20
         0.14770            29.537535          31.790352               28.081308           20
         0.17019            39.105156          41.877924               39.105156           20
         0.19610            52.487993          53.928300               52.487993           20
         0.22596            68.604465          69.344164               68.604465           20
         0.26036            88.108702          88.793835               88.108702           20
         0.30000           110.683200         112.525214              110.683200           20
