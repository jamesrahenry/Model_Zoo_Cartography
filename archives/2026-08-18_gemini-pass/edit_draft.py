import re

with open('paper/DRAFT.md', 'r') as f:
    text = f.read()

# Fix Abstract
text = text.replace("The headline is an exact law:", "We find that")
text = text.replace("invariant to width, class separation, and dataset, verified per-net with zero exceptions.", "invariant to width and class separation on controlled tasks, verified per-net with zero exceptions, and closely tracking this value on real data.")

# Fix Motivation
text = text.replace("The question this paper answers: **can structure measured from weights alone —\nbefore the network is ever run — predict where learned representation will\nappear once inference starts?** Concretely: given", "We investigate whether structure measured from weights alone can predict where learned representation will appear once inference starts. Given")
text = text.replace("Two research lines meet here, and neither can answer this alone.", "This work bridges two research lines.")
text = text.replace("The answer, developed in §4 and stated in full in §5, has three\nparts: weights alone carry the **where** (location, rank,\nand persistence of structure are weight-determined); the weight×input\ninteraction sharpens the **what** (the code's identity is real but\nrotation-hidden); and quantitative **how much** requires population-fitted\ncorrections.", "We find that weights alone determine the location, rank, and persistence of structure; the weight-input interaction establishes the code's identity (which remains rotation-hidden); and quantitative prediction requires population-fitted corrections.")

# Remove "Meaning:" blocks in Section 4
text = re.sub(r'\*Meaning:[^*]+\*\s*', '', text)

# Fix F1 scope
text = text.replace("L0 significant dims = C−1 under the fixed analytic floor: exact per-net", "Under the fixed analytic floor, L0 significant dims = C−1. This is exact per-net")

# Fix F2 scope
text = text.replace("### 4.2 The terminal rank is task-driven, two-sided", "### 4.2 The analytically-propagated terminal rank is task-driven")
text = text.replace("Trained\nnets' terminal rank tracks the task code instead", "Trained nets' analytically-propagated terminal rank tracks the task code instead")
text = text.replace("One law, signed by\nthe task's rank demand relative to the architecture's fixed point.", "This provides a weight-space structural signature, though it is computed using an uncorrected analytic chain known to carry significant error on trained weights (see §4.7).")

# Fix Discussion 5.1
text = re.sub(r'Can pre-inference structure predict where learned representation will appear\nonce inference starts\? The answer has three parts, each carried by a\ndifferent instrument:\n\n\*\*Weights carry the \*where\*\.\*\* ', 'Our findings indicate that ', text)
text = text.replace('**The weight×input interaction sharpens the *what* — and hides it in\ncoordinates.** ', '')
text = text.replace('**Quantitative *how much* needs population calibration.** ', '')

# Fix Discussion 5.2
text = text.replace("Inference is the end product — a model matters only when it runs — so it is\nworth being precise about what this paper claims can be known before it runs.", "It is important to delineate what can be known from weights prior to inference.")

# Remove LLM-isms
text = text.replace("Everything above is a matched-corpus result; models trained by other people,\nwith other optimizers, on other objectives, are a separate study. What we can\nreport is that the instruments survive the trip.", "While the above results rely on a matched corpus, preliminary tests indicate these instruments transfer to wild models.")

with open('paper/DRAFT_edited.md', 'w') as f:
    f.write(text)

