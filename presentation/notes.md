# Presentation notes — HADES v3 deck

Plain-English speaker notes for every slide, in order. Written to be read at a glance while you're
talking — each slide has: what to say in one breath, an analogy to fall back on if someone looks
confused, the one or two numbers worth saying out loud, and (where useful) a ready answer to the
obvious follow-up question. Nothing here requires a technical background to say confidently.

---

## Slide 1 — Title

**Say this:** We built a system that predicts three supply-chain problems — shipments running
late, products running out, and suppliers causing knock-on damage — and, just as importantly, it's
honest about which of its own answers it can actually back up with evidence.

**Analogy:** Think of a doctor who doesn't just say "you're at risk" — they say *why*, and they
tell you plainly when they genuinely don't know why, instead of making something up to sound
confident.

**Key number:** Everything was tested 25 different ways (5 different simulated "worlds" × 5
different random starting points) so a lucky one-off result can't sneak through.

---

## Slide 2 — The Finalised Architecture

**Say this:** The system has 5 layers, stacked like an assembly line. The first two layers just
make the prediction. The last three layers decide whether we're allowed to explain it, and how.

**Analogy:** Layers 1–2 are like a smoke detector — it just says "smoke detected." Layers 3–5 are
the fire inspector who shows up after and either says "here's exactly what caused it, verified,"
or "we don't actually know yet, so don't guess."

**Key line to land:** The core design rule is: if a part of the system fails its own test, we
throw its answer away rather than keep it because it *looks* smart. That one rule shapes
everything else in the deck.

---

## Slide 3 — Layer 1: SHARE, why it never changed

**Say this:** We tried seven different designs for the part of the model that reads the
supply-chain network. One design — we call it SHARE — was the best (or tied for best) at all
three predictions at once. Nothing else managed that.

**Analogy:** It's like auditioning seven employees for a job that actually has three different
sub-tasks. Six candidates were great at one or two of the sub-tasks but weak on the third. One
candidate was solidly good at all three — that's who got hired, permanently.

**Key numbers:** Current scores (AUC — think of it as "how good this is at telling risky cases
from safe ones," 1.0 is perfect, 0.5 is a coin flip): delay 0.78, shortage 0.91, impact 0.93.

**New this round:** Shortage's and impact's scores went up (shortage especially, 0.79 → 0.91) —
but not because we changed this part of the model. We changed the *map it reads*, not the
*reader*. Next slide explains what and why.

---

## Slide 4 — How Layer 1 works

**Say this:** Three ideas: (1) instead of giving every type of relationship in the network its own
private rulebook, they share a pool of common rules — so the rare relationship types can't
"memorize" instead of learning. (2) One shared "pay attention to this neighbor" mechanism flags
which connections matter. (3) It repeats this four times, so information can travel four steps
across the network.

**Analogy:** Imagine ten interpreters translating for eight different language pairs. Instead of
training ten completely separate interpreters (expensive, and the rare language pairs get too
little practice to be reliable), you train a shared pool of translation skills that any interpreter
can draw from. The rare pairs benefit from everything the common pairs have already learned.

**If asked "why focal loss / why fancy math":** Skip the equations — the plain version is "the
model repeats a simple update step four times, and each time it asks 'which of my neighbors should
I actually listen to?'"

---

## Slide 5 — Layer 2: Markov readout, why it never changed

**Say this:** Different questions need to look different distances into the network. "Will *this*
shipment be late?" only needs to look at itself. "Will a product run short?" needs to look one step
further out. "Will a supplier's problem ripple outward?" needs to look the furthest. So instead of
letting the model *learn* how far to look (which we tried, eight different ways, and it never beat
just fixing the distance), we fixed it directly, per question.

**Analogy:** If you want to know whether it'll rain in the next hour, you look outside your window.
If you want to know whether a hurricane will hit your coast next week, you look at a satellite
map covering a thousand miles. You don't need to *learn* which one to check — the nature of the
question already tells you.

**New this round — the big one:** We found the "shortage" prediction had a hidden problem: the
network map itself was unstable because one type of node (Carrier) had become a massive traffic
hub, and that instability was quietly canceling out a real, useful signal (live restocking
information) hiding right behind it. We fixed the map — split that one overloaded hub into many
smaller, well-organized ones — without touching any of the reading or scoring logic. Result:
**shortage passes its reliability bar for the first time ever in this project**, and impact's
pass, which had been knocked out by the same instability, came back too.

**Analogy for the fix:** Imagine one massive customer-service hotline number that every single
caller in the country dials, versus regional numbers organized by area code. Same total calls, same
information — but the regional setup doesn't jam up and drop calls the way one overloaded line
does.

**Also new:** We double-checked whether "how far to look" should change now that the map changed.
We tested other distances — none beat what we already had. So this part of the design is
re-confirmed, not just assumed.

---

## Slide 6 — How Layer 2 works

**Say this:** Once the network has processed everything, this layer does one thing: picks out the
right "zoom level" for each question (no learning involved, literally a lookup), then a tiny
scoring step turns that into a percentage.

**Analogy:** It's like a camera with three pre-set zoom buttons — "close-up," "medium," "wide" —
instead of an auto-zoom that has to guess every time. Pre-set buttons can't get confused; they
always zoom to exactly the right level for the shot you're taking.

**Why "zero parameters" is good, not lazy:** Nothing here can be memorized, drift over time, or
behave differently between two training runs. It's a deliberate design choice, not a shortcut.

---

## Slide 7 — Layer 2: are the probabilities honest? *(new)*

**Say this:** Getting the ranking right (who's riskiest) is one thing. Getting the actual
*percentage* right is a completely separate thing, and we checked it for the first time this
round. What we found: all three predictions were "shouting louder" than they should — for
example, shortage said there was a 36% chance on average, when the real rate was only about 5%. A
factor of nearly seven times too loud.

**Analogy:** Imagine a fire alarm that's correctly wired to go off exactly when there's smoke (so
it never misses a real fire — that's the "ranking" part working fine), but it also always
displays "90% chance of fire" even when the actual odds, historically, are more like 15%. It's
pointing at the right moments, it's just exaggerating the number every time.

**Why this happened — not a bug:** The way we trained the model deliberately balances rare and
common outcomes 50/50, on purpose, so the rare cases still get learned properly. The side effect
is the number it reports afterward is inflated in a very specific, predictable way. Because we knew
exactly what caused it, we could undo it with a formula instead of guessing.

**What we did:** Applied that formula (zero guesswork, no new data needed) to undo it — this alone
fixed 86–89% of the problem on all three predictions. Then we tried one extra small adjustment, but
only kept it where it passed a strict test on data it had never seen before; we threw it out for
one task (shortage) when it looked good on paper but quietly made things worse.

**Key numbers to say out loud:** Impact ended up the most improved — 93% of its original error is
gone. Shortage and delay: about 89% gone. And to be clear — the ranking (who's riskiest) was never
affected by any of this, only the honesty of the number.

**If asked "does this mean it's perfect now":** No — above a certain risk level, the number should
be read as "riskier than that one" rather than a literal percentage. But that's a small, clearly
marked slice of cases, and it doesn't affect which cases get flagged as highest priority.

---

## Slide 8 — The Layer 3 problem: detection is easy, explanation is not

**Say this:** Getting a good risk score is the easy 80%. The hard part is answering "why" in a way
you could actually defend to someone. We found four separate ways that a model can produce a
confident-sounding "reason" that is actually meaningless.

**Analogy:** A weather app can tell you "80% chance of rain" (detection) very reliably. But if it
then said "because a butterfly flapped its wings in Brazil," that explanation — even if it sounds
technical — isn't something you could act on or verify. Confident-sounding ≠ true.

**The four traps, in plain terms:**
1. The model blamed "how much time has passed" for a delay — but time passing is just a side
   effect of waiting, not a cause of anything.
2. One method always blamed the same single feature for over half of all impact cases — a
   fixed answer that never changes case-by-case isn't really "explaining" anything.
3. The real root causes are never directly visible in the data at all — only their downstream
   symptoms are. It's like only ever seeing footprints, never the animal.
4. Even the one method that pointed at a real, related factor still couldn't show that changing
   that factor would actually change the outcome.

**The takeaway line:** A number that looks important isn't the same thing as a number that's
*causing* the result. That distinction became the whole design philosophy from here on.

---

## Slide 9 — Layer 3: the evolution

**Say this:** This is the "we tried a lot of things, honestly" slide. Over multiple redesigns, we
kept finding real, promising signals that then fell apart under stricter testing — and each time,
instead of quietly dropping the negative result, we used it to redesign the next attempt.

**Analogy:** It's like prototyping a bridge design five times, and each time the bridge holds up
fine in mild weather but a stricter stress test finds a new weak point. Eventually you stop trying
to patch the same design and build something structurally different — which is what the final
version (a "validation gate stack," next slide) is.

**One thing worth saying plainly:** Every one of these was a genuine, real signal that just wasn't
strong enough yet — not a dead end caused by a mistake. That distinction matters for how confident
we should be about "not yet" versus "no."

---

## Slide 10 — The finalised Layer 3

**Say this:** Instead of trying to always produce an explanation, this layer's job became deciding
*whether* an explanation is trustworthy enough to say out loud. There are four checks, done in
order, and if an earlier one fails, we don't even bother running the later ones for that case.

**Analogy:** Think of airport security checkpoints — ID check, bag scan, pat-down, gate check.
If you fail the ID check, they don't bother scanning your bag; you're already not getting through.
Each gate here works the same way.

**Key line:** Two of the four checks failed for basically every case we tested. That's not treated
as an embarrassment — it's the entire point of building the checks: to stop an unproven "why" from
being said with false confidence.

**One subtlety worth mentioning:** For "shortage," one check couldn't even be run at all — that's
different from "we ran it and it failed," and we keep that distinction visible rather than
lumping them together.

---

## Slide 11 — How Layer 3 works

**Say this:** The technical heart of the checks: does changing something actually change the
outcome (tested by holding everything else identical and looking only at what's left over,
compared against a known-random baseline), and separately, are the proposed cause-and-effect
rules actually correct (checked by re-deriving them independently and comparing against the
system that generated the data).

**Analogy for the "residual" idea:** If you want to know whether a specific ingredient makes a
cake taste better, you don't just taste two random cakes — you bake two cakes that are identical
in every other way, change only that one ingredient, and see what's left of the taste difference
after accounting for everything else.

**Key number:** One promising-looking signal (0.55 out of 1.0) actually *failed* our bar, because
its result swung too much between different simulated scenarios. If we'd only run it once, we
would have wrongly called it a success.

**Mechanism check:** We re-verified 16 out of 16 proposed cause-and-effect rules by hand against
the actual source logic — and they held up. But a tempting four-step story connecting supplier
failure to eventual shortage was *rejected*, because one of its four links couldn't be verified —
three out of four correct steps still isn't a proven chain.

---

## Slide 12 — Layer 3: what's still wrong

**Say this:** Every "failed" result here failed because we don't have *enough data* to be sure —
not because the underlying signal is fake. That's an important and different kind of "not yet."

**Analogy:** It's like a clinical trial with only 5 patients — even a real medical effect can fail
to reach statistical significance with that few people. The effect might be completely real; you
just don't have enough evidence yet to say so confidently. More patients (here: more real-world
data) is the fix, not a different drug.

**Key honest line for the room:** One check only works right now *because* we built the data
ourselves and know its exact rules in advance. In the real world, nobody hands you the answer key
— so this is simultaneously our strongest result and the part least likely to transfer as-is.

---

## Slide 13 — The finalised Layer 4

**Say this:** Every prediction the system makes comes packaged with six separate pieces of
information, and — critically — none of them is allowed to stand in for another one.

**Analogy:** Think of a food nutrition label. Calories, sugar, protein, and allergens are all
separate numbers on purpose — a company can't just report "healthy: yes/no" and call it equivalent.
Each fact stays its own fact.

**Key detail:** If a piece of evidence isn't there, the system says "Unsupported" out loud instead
of just leaving that field blank. A blank field could be mistaken for "not relevant here"; the
truth is "we don't have proof of this anywhere." That's a deliberate, important difference.

---

## Slide 14 — How Layer 4 works

**Say this:** The system has hard, automatic rules that block certain fields from ever being
confused with each other — for example, a raw "this seems important" score can never sneak into
the box labeled "this is a proven cause." These aren't style guidelines; the software actively
rejects it if someone tries.

**Analogy:** It's like a hospital chart where "the nurse's gut feeling" and "the confirmed lab
result" are printed in physically different sections of the form, in different colors, so a
rushed doctor can never mistake one for the other.

**One nuance worth explaining if asked:** Confidence has two sources of "wobble" — how much
answers vary if you retrain the exact same model, versus how much they vary across genuinely
different scenarios. The second source turned out to be by far the bigger one (up to 99% of the
total wobble) — so measuring only the first kind, which is what most systems do, would have been
looking at the smaller, less important source of uncertainty the whole time.

---

## Slide 15 — Layer 4: what's still wrong

**Say this:** A few of the six fields are permanently stuck showing "Unsupported" right now — not
because the system is broken, but because they require a specific kind of real-world evidence
(actual past decisions and their outcomes) that a simulated dataset structurally cannot provide.

**Analogy:** It's like a form that asks "what did the customer decide after seeing this
recommendation?" — you literally cannot fill that field in unless the recommendation was actually
shown to a real customer at some point. No amount of clever modeling substitutes for that.

**Key reassurance for the room:** The report card format itself — the six-field structure — is
already finished and ready to use. Swapping in real data changes the *answers* on the report card,
not the report card's design.

---

## Slide 16 — The finalised Layer 5

**Say this:** What the system is *allowed to say* — plain monitoring language versus a full
recommendation to act — depends entirely on which checks from Layer 3 actually passed. It is
never based on how confident the model feels, only on what's been proven.

**Analogy:** It's like a doctor who won't prescribe a specific treatment just because a patient's
symptoms look severe — a specific treatment recommendation requires an actual diagnosis, not just
a worried-looking patient. Severity and certainty are different things, and mixing them up is
exactly the mistake this layer prevents.

**Key line:** A high-confidence number with no proven cause behind it still only gets to say
"monitor this" — never "do this because it will fix that." Confidence about the *number* is not
the same as confidence about *why*.

---

## Slide 17 — How Layer 5 works

**Say this:** The wording for each confidence tier is locked in ahead of time, not improvised in
the moment — so a stronger claim can never accidentally leak into a weaker one's message. There's
also an automatic language filter that blocks words like "causes," "reduces," or "prevents"
anywhere the evidence hasn't earned them.

**Analogy:** It's like a legal disclaimer team reviewing marketing copy before it goes out — certain
words ("cures," "guarantees") are banned outright unless specific evidence backs them, regardless
of how the writer feels about the product.

**Worth reading out loud, verbatim, if you have a minute:** The example message on the slide
("MONITOR. delay risk... No validated explanation is available...") is the actual output format.
It's a good moment to pause and just read it — it makes the whole idea concrete.

---

## Slide 18 — Layer 5: what's still wrong

**Say this:** Right now the system almost always lands in the most cautious tier — not because the
higher tiers don't work (they're built and tested), but because the evidence needed to unlock them
doesn't exist yet in a simulated world.

**Analogy:** It's like having a promotion ladder fully designed and posted on the wall, but nobody
has had the chance to actually earn the qualifications yet because the training program only just
opened.

---

## Slide 19 — The architecture end to end

**Say this:** This is one real example, start to finish, for the single riskiest shipment in our
test data — showing exactly which checks it passed, which it failed, and what the system was
allowed to say about it as a result.

**Analogy:** Think of this as a single patient's full chart, start to finish — test results,
which diagnoses were confirmed vs. ruled out, and the final, appropriately careful next step. It's
the "here's a real one, walked through" slide, which tends to land better than any summary table.

**Key detail to point out:** Two explanations that a less careful system would have confidently
offered were specifically caught and rejected here — a good, concrete way to show the safety net
actually catching something, not just existing in theory.

---

## Slide 20 — How the architecture solves the problem

**Say this:** The core insight is that most systems mash four different questions into one
number — how likely, how confident, why, and what to do. We deliberately kept them separate, each
with its own test, so a strong answer to one can never be mistaken for a strong answer to another.

**Analogy:** It's like separating "the patient is very sick" from "we know exactly what's wrong"
from "we're sure this treatment will help." Conflating those three in an emergency room would be
dangerous — yet a single risk score often quietly conflates all of them.

**Best line to just say directly:** "No validated explanation is available, and here's exactly
which test it failed" is genuinely more useful than a confident-sounding guess — because you can
act on it (get more evidence) instead of being misled by it.

---

## Slide 21 — This is not a supply-chain architecture

**Say this:** Swap out "shipments" for patients, transactions, or equipment, and this exact
pattern — good detection, hard-to-prove explanation — shows up everywhere high-stakes decisions
get made from data.

**Analogy:** The design is like a universal seatbelt mechanism — the specific car model changes,
but the click-and-lock mechanism protecting the passenger works the same way regardless. Swap the
"car" (the industry), keep the "seatbelt" (the validation discipline).

**Use this if the room asks "so is this just for supply chains":** No — call out one example
relevant to your audience (healthcare, finance/fraud, equipment maintenance are all on the slide)
and note the same four gates would guard the same four questions there too.

---

## Slide 22 — Why this is a benchmark, not just a model

**Say this:** Most published results report a single number and stop. We report the number *and*
the bar it had to clear to count as real — and several results that would have been published
elsewhere as wins were something we caught and threw out ourselves.

**Analogy:** It's the difference between a runner claiming a personal best on a single favorable-
wind day, versus a track federation that only certifies records set under strict, wind-checked
conditions. The second standard is harder to meet, but the records that do clear it actually mean
something.

**Key line:** Turning "did we over-claim?" into an automatic, code-level test — rather than a
matter of writing carefully — is the actual contribution here, separate from any one prediction
result.

---

## Slide 23 — What's still missing, and why only real data closes it

**Say this:** Every single "not yet" in this whole project traces back to one root cause: we're
working with a simulated world of a realistic but limited size. None of them trace back to a flaw
in the design.

**Analogy:** It's like a new medical test that works perfectly in a lab with 5 test samples — the
science is sound, but you'd never approve it for hospitals without testing it on thousands of real
patients first. Small-scale proof and real-world scale are different milestones, not different
goals.

**Good closing line for the meeting:** We know exactly what a real dataset would need to contain
to close each remaining gap, and we've already gone looking for one — that's a stronger, more
credible place to end on than pretending everything is already finished.

---

## If you only have 5 minutes

Hit these four beats, in order:

1. **What it does:** predicts three supply-chain risks, and — new this round — those predictions
   are now honest, not just accurate (slide 7's "shouted 36%, meant 5%" example is the easiest
   thing in the whole deck to make people nod along to).
2. **What changed this round:** one prediction (shortage) passes its reliability bar for the first
   time ever, because we found and fixed an overloaded hub in the network map — not because we
   changed the model itself.
3. **Where the system draws its own line:** it refuses to state a "why" it can't back up, and says
   so plainly instead of guessing.
4. **Where it's headed:** every open gap is a data-scale problem with a known fix, not a design
   flaw.

## Anticipated questions

**"So is it done?"** — The prediction layer is now in a strong, honestly-measured state on two of
three tasks (shortage, impact) and a weaker-but-understood state on the third (delay). The
explanation layer (Layer 3+) is deliberately conservative until it has more evidence to work with.

**"Why does shortage's pass only 'barely' clear the bar?"** — Say it plainly, don't dodge it: the
margin is thin (about 2% of the required buffer). It's a real pass under the agreed rule, not a
comfortable one, and that's worth stating exactly as measured rather than rounding it up.

**"Can we trust the percentages it gives us?"** — Yes, for the large majority of cases (89–99.7%
depending on which prediction), the stated percentage is now close to the true rate. Above a
task-specific threshold, treat the number as a ranking signal rather than a literal percentage —
and that threshold, and the reason for it, is fully documented.
