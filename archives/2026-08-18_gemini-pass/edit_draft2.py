import re

with open('paper/DRAFT_edited.md', 'r') as f:
    text = f.read()

# Abstract fixes
text = text.replace("The headline is an exact law: the input weight", "We find that the input weight")
text = text.replace("invariant to width, class separation, and dataset, verified per-net with zero exceptions.", "invariant to width and class separation on controlled mixtures, verified per-net with zero exceptions (and closely tracking C-1 on real data).")

# F1 real data clarity
text = text.replace("8.25–9.4 on real data;", "reads 8.25–9.4 on real data (closely tracking C−1 but not exact per-net);")

# Remove extra LLM phrases
text = text.replace("Three results carry over:", "We identify three implications:")
text = text.replace("What we can report is that", "Preliminary tests indicate that")

# F2 deeper scope fix
# DRAFT.md has: "Trained nets' analytically-propagated terminal rank tracks the task code instead"
# Let's ensure there are no lingering misconceptions about it being the literal rank.
text = text.replace("the rank the code carries at the last layer is set by the task", "the analytically-propagated rank at the last layer is set by the task")

# Section 5.1
text = text.replace("Our findings indicate that Location, rank", "Location, rank")

with open('paper/DRAFT_edited2.md', 'w') as f:
    f.write(text)

