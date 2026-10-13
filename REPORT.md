# Assignment 1 Report

- **Name**: Bernarda Andrade
- **Student ID**: 19119
- **Email**: bandrade.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

I used the *Speed Dating & Attraction Outcomes* dataset by sergionefedov on
Kaggle, a synthetic dataset modelled on the Columbia speed-dating
experiment by Fisman & Iyengar (2002–2004). Because it is synthetic, I understand that it does not
describe real people, so any patterns found can help show whether the methods used can predict an outcome. Each row
is a four minute date between a man and a woman at a speed dating event, with
both people's ages, whether they share a race or field, a shared interests
score, the ratings each gave the other (attractiveness, sincerity,
intelligence, fun, ambition), their stated preferences and self ratings, how
often each goes out, and each person's yes/no decision. The file has 14,504
rows and 30 columns covering 120 events and 2,548 participants (each attends
one event), with no missing values. The target, `match`, is 1 when both people
said yes. I chose this dataset because I find
it interesting to analyse human attraction, especially to find out whether it
is possible to predict outcomes like a mutual match.

## Business / real-life framing

This model helps a speed dating organiser. During each event every participant writes a scorecard that rates each partner. Many participants, however, only send their yes/no answers days later, or they never send them. The morning after the event, the organiser uses the model to decide which pairs are likely to match mutually. The organiser then sends a "you two should talk" nudge to those pairs while interest is still fresh. Because the model makes predictions after the dates, the model can use both profile information (age, field of study, race, stated preferences, self ratings, how often each person goes out) and the scorecard ratings each person gave their partner. The question the model tests is whether a mutual match can be predicted from how the date went, not only from who the people are. I expected a real but moderate signal.

**Target.** The target is `match`, which is set to 1 when both participants said yes. I dropped `male_decision` and `female_decision` because `match` is computed directly from those two decisions, so keeping them would leak the answer into the features.

**Cost of errors.** A false positive is a nudge sent to two people who would not have matched. This is mildly annoying and may lead users to ignore future suggestions. A false negative is a real match that never reconnects, which is exactly the outcome the organiser's service tries to create. Therefore I assume a missed match costs three times as much as a wasted nudge, so business cost equals FP × 1 + FN × 3. I chose the decision threshold that minimises this cost on the validation set instead of using the default 0.5.

**Metric for comparing models.** To compare models independently of the threshold, I used ROC‑AUC and PR‑AUC. PR‑AUC is especially suitable because matches are relatively rare (19.6% of dates). Accuracy would be misleading because a model that always predicts "no match" already scores about 80%.

**Split.** I did not use a time based split. The data has no dates and `event_id` is only an identifier, so there is no "future" that could leak into training. Instead I split by `event_id`, keeping whole events together in train, validation or test. Each participant has 6–15 dates (11 on average) within a single event. A random split would place the same person in both train and test. The model could then memorise individuals instead of learning general patterns, which would inflate the test score.

## Data preparation & feature engineering

**Data quality and leakage.** The file has no missing values, so nothing was
imputed, and I found no defective segments, so no rows were excluded. I
verified that `match` equals `male_decision` AND `female_decision` on every
row, then dropped both decision columns, since keeping them would hand the
model the answer. That left 24 candidate features. The ID columns are kept
aside to build the split and are never model inputs. Every choice below was
made with 5-fold cross-validation grouped by `event_id` on the 84 training
events (validation and test were not used), and the number to beat is the
naive baseline (CV PR-AUC 0.198, i.e. the match rate). All features are
standardised with a scaler fitted on the training split only.

**Decision 1: "Both interested" features (dropped).** A match needs both
people to say yes, so I added `min` and product features of the two people's
ratings, plus `overall_by_male` and `overall_by_female` (the average of the
five scorecard ratings each person gave the other). I expected them to help
because logistic regression combines its inputs by adding them. Grouped CV
showed no gain (0.397 vs. 0.397 for the base set), so I dropped them. The
reason is that logistic regression doesn't add up probabilities; it adds up
log-odds. A match happens when he says yes *and* she says yes, so
P(match) ≈ P(he says yes) × P(she says yes), and multiplying probabilities is
the same as adding their logarithms. An additive model of the two people's
ratings therefore already captures "both must agree."

**Decision 2: Rater calibration (dropped).** Some people rate everyone a 9, so
I compared each rating with the rater's own average, such as "she rated him 2
points above her usual." This is allowed in my scenario because all of a
person's scorecards exist the morning after, and the averages use ratings
only, never decisions. I expected calibration to help, but grouped CV showed
it slightly lowered PR-AUC (0.394 vs. 0.397), so I removed it. One possible
reason is that with only 6–15 dates per person, the personal average is noisy.

**Decision 3: Preference-weighted ratings (kept).** I multiplied how much a
person says they value attractiveness (and intelligence) by how attractive
(and intelligent) they rated their partner. Stated priorities often predict
real choices poorly, so I treated this as an experiment and kept it only if it
improved CV PR-AUC. It improved it from 0.397 to 0.403, so I kept it. This
was the only engineered group that helped.

**Decision 4: `goes_out` encoding (ordinal).** The categories have a natural
order (rarely = 0 to very often = 3), so one ordered column keeps that
information with fewer features than one-hot encoding. CV showed no clear gain
from one-hot (0.396 vs. 0.397), so I kept the ordinal version.

**Decision 5: Profile features (kept).** Age, race, field and preferences are
almost uncorrelated with matches: on their own they reach only 0.237 CV PR-AUC
(raw ratings alone: 0.394). But removing them from the final set lowers CV
PR-AUC from 0.403 to 0.394, so they matter in combination. The model uses them
as context for interpreting the ratings: stated preferences get negative
coefficients alongside the positive preference-weighted features, so the same
rating counts differently depending on how much the person says they care
about that attribute.

Each engineered group was compared with the same base set (profile + raw
ratings) and kept only if it raised mean CV PR-AUC. The final model uses 28
features: the base set plus the four preference-weighted features.

| Variant (grouped 5-fold CV, training events) | Features | CV ROC-AUC | CV PR-AUC (± std) |
|---|---|---|---|
| Naive baseline | 0 | 0.500 | 0.198 ± 0.006 |
| Profile features only | 13 | 0.570 | 0.237 ± 0.017 |
| Raw ratings only | 11 | 0.716 | 0.394 ± 0.008 |
| Base: profile + raw ratings | 24 | 0.717 | 0.397 ± 0.009 |
| Base + both-interested (dropped) | 34 | 0.718 | 0.397 ± 0.008 |
| Base + rater calibration (dropped) | 37 | 0.715 | 0.394 ± 0.009 |
| **Base + preference-weighted (final)** | **28** | **0.722** | **0.403 ± 0.007** |
| Base, `goes_out` one-hot instead of ordinal (rejected) | 30 | 0.717 | 0.396 ± 0.009 |
| Final set without profile features | 15 | 0.716 | 0.394 ± 0.008 |

## Modeling: three implementations, one model

*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*

**Do the three agree?** Yes. On the final 28 features, the three
implementations reach the same training objective (0.44354), their
coefficients differ by at most 0.001 and their test probabilities by at most
0.0005. An earlier version with the "both interested" features did not agree
as well: `overall_by_male` and `overall_by_female` are exact averages of five
raw ratings that were also features (perfect multicollinearity), so many
coefficient combinations gave the same predictions, and gradient descent
approached the regularised optimum much more slowly along that flat direction
than scikit-learn's solver (coefficients differed by up to 0.12). Dropping
that group removed the problem.

**Why Standard PyTorch has a different confusion matrix.** Although its
probabilities match the other two almost exactly, its cost-minimising
validation threshold came out at 0.22 instead of 0.29. The validation cost
curve is very flat between those values (891–894 vs. 893), so a 2-point
difference decides which threshold "wins." On the test set this changes the
cost from 841 to 898. The difference comes from threshold selection on a
small validation set (18 events), not from the model.

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*
