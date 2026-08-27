# Background

[Back to the README](../README.md)

---

Why collaboratively tuning a Dialogue State Tracking model without a central server is worth doing, and why DST is a good flagship task for WAFL.

WAFL is a way of tuning a model **for a local community**, using nothing but the
device-to-device contacts that the community's own movement produces. That is a
convenient property: nobody has to stand up a training service, register the
devices with it, keep it running, or agree on who operates it. A group of
devices that happen to be in the same place — a campus, a hospital, a museum, a
shopping street, a factory floor — can improve their models simply by meeting
each other.

Dialogue State Tracking is a natural flagship LLM application for this idea.

A task-oriented dialogue assistant only works if it can convert what a person
just said into a structured state: which domain they are talking about, which
slots they have filled, and with what values. What that state space looks like
in practice is intensely local. The venues people ask for, the phrasing they
use, the services that exist, the way a booking is described, and the slots that
actually matter differ from one community to the next. A model tuned once,
centrally, on generic data is generic everywhere; a model tuned to a community
is useful in that community.

At the same time, no single device in a community ever sees the whole picture.
One device accumulates restaurant bookings, another mostly train connections,
another taxi rides. Tuned alone, each of them becomes a narrow specialist and
gets worse — not better — at everything else, which is exactly the Non-IID
problem WAFL was designed for. Collaborative tuning is what lets every device
end up with a model that covers the community's whole domain and slot space,
not just the part it happened to observe.

DST is also a good testbed, for concrete reasons:

* **The output is structured and objectively scorable.** Joint Goal Accuracy is
  exact-match on a JSON object, so there is no room for judging quality by feel.
* **The data partitions naturally.** MultiWOZ dialogues carry an inherent domain
  label, which gives a realistic Non-IID split rather than an artificial one.
* **It is small enough to actually run.** With LoRA on a 4-bit base model, ten
  simulated nodes fit on a single consumer GPU.

---
