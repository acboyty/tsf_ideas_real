no use - feature - gate

sin_mul (0.01, 5) - (0.005, 10) - (0.005, 5, 5)
MAE 0.01530 MSE 0.02080 - MAE 0.01437 MSE 0.03624 - MAE 0.00951 MSE 0.01103
MAE 0.00573 MSE 0.00494 - MAE 0.01315 MSE 0.02780
MAE 0.00354 MSE 0.00131 - MAE 0.00747 MSE 0.02715

prophet (20-1) (28-7) (bfill) (bfill holiday lshift 1)
MAE 0.33633 MSE 0.55073 - MAE 0.40334 MSE 0.64954 - MAE 0.24181 MSE 0.15291 - MAE 0.24181 MSE 0.15291
MAE 0.34323 MSE 0.54814 - MAE 0.40793 MSE 0.65465 - MAE 0.23975 MSE 0.14643 - MAE 0.24145 MSE 0.14944
MAE 0.32729 MSE 0.52583 - MAE 0.41216 MSE 0.67254 - MAE 0.23551 MSE 0.14618 - MAE 0.23315 MSE 0.14065

0.1 - 0.05 - 0.03 - 0.01 - 0.005
no use  MAE 0.13013 MSE 0.23767 - MAE 0.05601 MSE 0.09517 - MAE 0.04165 MSE 0.07393 - MAE 0.01865 MSE 0.02492 - MAE 0.00872 MSE 0.00985
feature MAE 0.00633 MSE 0.00029 - MAE 0.00468 MSE 0.00042 - MAE 0.00560 MSE 0.00118 - MAE 0.00634 MSE 0.00339 - MAE 0.00635 MSE 0.00636
gate    MAE 0.03038 MSE 0.00644 - MAE 0.01452 MSE 0.00539 - MAE 0.01113 MSE 0.00476 - MAE 0.00475 MSE 0.00083 - MAE 0.00234 MSE 0.00015

--------------------------------------------------------------

0.1 - 0.05 
no use              MAE 0.14846 MSE 0.26007 - MAE 0.06678 MSE 0.09485
feature             MAE 0.00769 MSE 0.00059 - MAE 0.00642 MSE 0.00065
gate withholiday    MAE 0.02142 MSE 0.00327 - MAE 0.01673 MSE 0.00328
gate withoutholiday MAE 0.00040 MSE 0.00000 - MAE 0.00036 MSE 0.00000
feature gate        MAE 0.00434 MSE 0.00006 - MAE 0.00360 MSE 0.00006

--------------------------------------------------------------

effect duration == 5
0.05 - 0.03 - 0.01
no use              MAE 0.12426 MSE 0.23569 - MAE 0.07878 MSE 0.13735 - MAE 0.02613 MSE 0.03670
feature             MAE 0.07485 MSE 0.11395 - MAE 0.04634 MSE 0.07298 - MAE 0.02516 MSE 0.03306
gate withholiday    MAE 0.08525 MSE 0.29512 - MAE 0.04603 MSE 0.14942 - MAE 0.00776 MSE 0.00143
gate withoutholiday MAE 0.04487 MSE 0.08483 - MAE 0.02381 MSE 0.06385 - MAE 0.00072 MSE 0.00000
feature gate        MAE 0.04084 MSE 0.03113 - MAE 0.02654 MSE 0.02596 - MAE 0.00775 MSE 0.00144
> feature+gate seems to be the best except 0.01.
> gate with/withoutholiday performs better when holiday gets sparser.

holiday effect -> +
0.03 - 0.01            
no use              MAE 0.12058 MSE 0.37454 - MAE 0.02740 MSE 0.06093
feature             MAE 0.04614 MSE 0.09162 - MAE 0.01444 MSE 0.01733
gate withholiday    MAE 0.08331 MSE 0.14690 - MAE 0.02081 MSE 0.01374
gate withoutholiday MAE 0.13973 MSE 0.20904 - MAE 0.00903 MSE 0.01065
feature gate        MAE 0.04451 MSE 0.04192 - MAE 0.01492 MSE 0.01061
> feature seems better in + cases

-----------------------------------------------------------------------

mul plus gaussian

0.03 - 0.01 - 0.005 - 0.005(smaller gaussian)
no use          MAE 0.20620 MSE 0.35385 - MAE 0.07591 MSE 0.11972 - MAE 0.02810 MSE 0.04674 - MAE 0.02169 MSE 0.02555
feature         MAE 0.16357 MSE 0.21619 - MAE 0.06353 MSE 0.08358 - MAE 0.02812 MSE 0.04659 - MAE 0.02146 MSE 0.02633
gate            MAE 0.16361 MSE 0.21758 - MAE 0.06454 MSE 0.09138 - MAE 0.02558 MSE 0.04453 - MAE 0.01770 MSE 0.00653
feature gate    MAE 0.16063 MSE 0.21495 - MAE 0.05868 MSE 0.08192 - MAE 0.02570 MSE 0.04361 - MAE 0.01272 MSE 0.00305

-----------------------------------------------------------------------

mul 10w dots

no use      MAE 0.01260 MSE 0.01702
feature     MAE 0.00842 MSE 0.01218
gate        MAE 0.00279 MSE 0.00074

-----------------------------------------------------------------------

mul decreasing

0.005(effect 10) - 0.005(effect 5) - 0.005(effect 5)
no use      MAE 0.14092 MSE 0.54907 - MAE 0.08721 MSE 0.36291 - MAE 0.02048 MSE 0.05695
feature     MAE 0.11716 MSE 0.37262 - MAE 0.07368 MSE 0.27280 - MAE 0.01692 MSE 0.03319
gate        MAE 0.12548 MSE 0.44210 - MAE 0.06092 MSE 0.15432 - MAE 0.01518 MSE 0.02959

------------------------------------------------------------------------

mul plus decreasing

0.005(effect 5) - 0.005(w more stable)
no use      MAE 0.04634 MSE 0.23095 - MAE 0.05038 MSE 0.30250
feature     MAE 0.03792 MSE 0.13801 - MAE 0.02314 MSE 0.03383
gate        MAE 0.03514 MSE 0.08863 - MAE 0.03285 MSE 0.19197
2gate       MAE 0.03415 MSE 0.08655 - MAE 0.03427 MSE 0.19483

------------------------------------------------------------------------

x[1..5] after holiday are computed as x[1..5] * linspace(5, 1, 5) + linspace(0.2, 0.02, 5)

holiday rate    0.01                        0.01 retrain                     0.01 regen                   0.005                      0.01 reregen

no use          MAE 0.11770 MSE 0.96767     MAE 0.11957 MSE 0.96741                                  MAE 0.05951 MSE 0.46842    
feature         MAE 0.01733 MSE 0.01571     MAE 0.01340 MSE 0.00778     MAE 0.01010 MSE 0.00375      MAE 0.01621 MSE 0.02899         MAE 0.01018 MSE 0.00675
gate            MAE 0.02961 MSE 0.09642     MAE 0.01887 MSE 0.03647                                  MAE 0.00594 MSE 0.00046
gate x & +      MAE 0.03569 MSE 0.15534     MAE 0.02325 MSE 0.07071     MAE 0.00683 MSE 0.00044      MAE 0.00527 MSE 0.00059         MAE 0.01515 MSE 0.13577

----------------------------------------------------

x after holiday are computed as x * 3 + 1

0.01 - 0.005

feature     MAE 0.00333 MSE 0.00026 - MAE 0.00209 MSE 0.00103
gate x & +  MAE 0.00399 MSE 0.00180 - MAE 0.00120 MSE 0.00001

-----------------------------------------------------------------------

> Conclusion: when holiday effects are known, then we can design the specific gate to beat feature(when holiday are really sparse).

--------------------------------------------------------------------

T1D

540

bolus - meal

no use      MAE 16.59874 MSE 507.71796 - MAE 16.59874 MSE 507.71796
feature     MAE 16.23643 MSE 491.83616 - MAE 16.71275 MSE 513.95362
gate        MAE 16.01501 MSE 481.92832 - MAE 16.72186 MSE 525.75671

544

bolus - meal

no use      MAE 13.43239 MSE 382.67123 - MAE 13.43239 MSE 382.67123
feature     MAE 12.45497 MSE 328.36496 - MAE 12.55543 MSE 330.87981
gate        MAE 12.60553 MSE 345.79648 - MAE 12.63608 MSE 341.30488

552

bolus - meal

no use      MAE 12.81945 MSE 305.94600 - 
feature     MAE 12.08993 MSE 268.50486 - MAE 12.85831 MSE 308.00239
gate        MAE 12.61063 MSE 275.93186 - MAE 12.83902 MSE 305.62679

-------------------------------------------------------------------
retail-data-analytics
no use      MAE 0.08015 MSE 0.01359
feature     MAE 0.08069 MSE 0.01347
gate        MAE 0.08243 MSE 0.01412

-----------------------------------------------------------
bike-sharing
no use      MAE 0.09423 MSE 0.02139
feature     MAE 0.09279 MSE 0.02081
gate        MAE 0.10358 MSE 0.02315

-----------------------------------------------------------
NYC Uber Pickups
Bronx_df - Queens_df - Brooklyn_df - Manhattan_df
no use      MAE 0.04641 MSE 0.00379 - MAE 0.05566 MSE 0.00551 - MAE 0.02931 MSE 0.00165 - MAE 0.02875 MSE 0.00185
feature     MAE 0.04638 MSE 0.00379 - MAE 0.05566 MSE 0.00553 - MAE 0.02923 MSE 0.00164 - MAE 0.02869 MSE 0.00185
gate        MAE 0.04690 MSE 0.00388 - MAE 0.05637 MSE 0.00562 - MAE 0.02931 MSE 0.00164 - MAE 0.02898 MSE 0.00187

-----------------------------------------------------------
shunfeng
sample 50 - sample 200 - sample 50 10fea
no use      MAE 0.14933 MSE 0.04387 - MAE 0.15074 MSE 0.04766 - MAE 0.18630 MSE 0.06193
feature     MAE 0.15921 MSE 0.05193 - MAE 0.16676 MSE 0.05748 - MAE 0.22166 MSE 0.08376
gate        MAE 0.16348 MSE 0.05970 - MAE 0.17506 MSE 0.06601 - MAE 1.44039 MSE 7.21549
fea+gate    MAE 0.16475 MSE 0.06576 - MAE 0.18085 MSE 0.07149 - MAE 0.94257 MSE 2.66965
prophet     MAE 0.40350/0.18974

-----------------------------------------------------------
london-bike
no use      MAE 0.02639 MSE 0.00270
feature     MAE 0.02638 MSE 0.00270
gate        MAE 0.02591 MSE 0.00261
fea+gate    MAE 0.02593 MSE 0.00261

-----------------------------------------------------------
prophet
no use      MAE 0.24181 MSE 0.15291
feature     MAE 0.23975 MSE 0.14643
gate        MAE 0.23551 MSE 0.14618  