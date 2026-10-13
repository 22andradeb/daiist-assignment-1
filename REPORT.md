# Assignment 1 Report


- **Name**: Bernarda Andrade
- **Student ID**: 19119
- **Email**: bandrade.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

## Dataset

I used the *Speed Dating & Attraction Outcomes* dataset by sergionefedov on
Kaggle, a synthetic dataset modelled on the Columbia speed-dating
experiment by Fisman & Iyengar (2002–2004). Because it is synthetic, I understand that it will not 
describe real people so any patterns foudn can help find if the methods used can help predict an outcome. Each row
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

This model helps a speed dating organiser. During each event every participant writes a scorecard that rates each partner. Many participants however only send their yes/no answers days later or they never send them. The morning the organiser uses the model to decide which pairs are likely to match mutually. The organiser then sends a "you two should talk" nudge to those pairs while interest's still fresh. Because the model makes predictions after the dates the model can use both profile information age, field of study, race stated preferences, self ratings how often each person goes out—and the scorecard ratings each person gave their partner. The question the model tests is whether a mutual match can be predicted from how the date went, not from who the people are. I expected a but moderate signal.

**Target.** The target is `match` which is set to 1 when both participants said yes. I dropped `male_decision` and `female_decision` because `match` is computed directly from those two decisions so keeping them would leak the answer into the features.

**Cost of errors.** A false positive is a nudge sent to two people who would not have matched. This is mildly annoying. May lead users to ignore future suggestions. A false negative is a match that never reconnects, which is exactly the outcome the organisers service tries to create. Therefore I assume a missed match costs three times much as a wasted nudge so business cost equals FP × 1 + FN × 3. I chose the decision threshold that minimises this cost on the validation set of using the default 0.5.

**Metric for comparing models.** To compare models of the threshold I used ROC‑AUC and PR‑AUC. PR‑AUC is especially suitable because matches are relatively rare 19.6% of dates. Accuracy would be misleading because a model that always predicts "no match" already scores about 80%.

**Split.** I did not use a time‑based split. The data has no dates and `event_id` is an identifier so there is no "future" that could leak into training. Instead I split by `event_id` keeping events together in train, validation or test. Each participant has 6–15 dates, 11, on average within a single event. A random split would place the person in both train and test. The model could then memorise individuals of learning general patterns, which would inflate the test score.

## Data preparation & feature engineering

*What you engineered and why, and any data-quality decisions you made along
the way — e.g. "segment X had defective data, so I excluded it and used a
population-average default for scope Y at inference time; the impact of
that choice is Z."*

## Modeling: three implementations, one model

*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*
