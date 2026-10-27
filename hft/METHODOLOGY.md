# Research & Development Methodology

*A high-level account of the process, the thinking, and the engineering discipline behind this system. Written to communicate approach and rigour — not to convey implementation specifics.*

---

## The Starting Point

The project began not with a strategy, but with a question about time. The first prompt was essentially: *does the same structure behave differently depending on when in the day it appears?* That question — about the interaction between a signal class and a temporal envelope — shaped everything that followed. It is still the animating question.

The answer, provisionally, is yes. Dramatically so. But arriving at a defensible answer required building infrastructure capable of asking the question properly — which turned out to be most of the work.

---

## How the Stack Was Built

The system spans two environments that must remain tightly coupled: a proprietary execution environment with hard constraints on what code can run inside it, and a general-purpose analytical environment where there are no such constraints. The core discipline was keeping these two worlds cleanly separated — the execution environment does as little as possible beyond taking actions and emitting structured records; every inference, attribution, and parameter decision lives in the outer environment where it can be tested, visualised, and revised without risking a live run.

Inside the execution environment, the architecture went through several iterations. Early versions mixed concerns freely — detection logic, execution logic, and risk logic were entangled. The mature version is layered: a recognition layer identifies candidate structures; a gate layer filters candidates against a set of market-state conditions; a scoring layer ranks survivors; and only then does an execution layer act. Each layer has its own configuration surface. Changes to one layer can be evaluated in isolation.

Outside the execution environment, a Python stack handles everything from raw data ingestion to backtesting to parameter search to machine learning to a real-time dashboard. The components are modular and compose — the same data loader feeds the backtest replay, the sweep runner, and the ML training pipeline.

---

## The Testing Philosophy

The most important methodological decision was to run in shadow mode before committing to live outcomes. The system was designed to emit a complete record of every candidate it considered — not just the ones it acted on — along with its internal state at the moment of evaluation. This record is the primary object of analysis. It decouples *what the system sees* from *what the system does*, which matters enormously when you are still in the process of understanding what the system sees.

A secondary testing layer runs dozens of execution-management variants in parallel on every candidate that passes the gate. Each variant applies a different management scheme to the same underlying entry. This is not a simulation — the variants operate on real market data in real time. The result is a continuous, live experiment that generates comparative data across variant space without requiring the user to commit capital to any single parameterisation. When one variant consistently outperforms others across a range of conditions, that becomes the next candidate for the live configuration.

Backtests were run across multiple non-overlapping time windows. The outputs — both aggregate metrics and granular per-segment breakdowns — are stored as structured artefacts so the evolution of findings can be traced. The window-split approach makes it possible to distinguish whether an improvement holds out-of-sample or whether it was fitted to a specific market regime.

Parameter searches are not exhaustive grid searches. They are hypothesis-driven: a specific range is swept because there is a reason to believe the optimum lives there, and the sweep result either confirms the hypothesis or reveals where to look next.

---

## How Analysis Was Approached

The rule that emerged early and held throughout: *never report an aggregate without also reporting what it is made of.*

Topline numbers are almost always misleading. A strategy that appears marginally positive in aggregate can be sharply negative in one subclass and strongly positive in another — and those two effects cancel, hiding the structure entirely. The habit of immediately slicing by every relevant dimension simultaneously — direction, structure type, time-of-day regime, and signal quality marker — is the only reliable way to surface what is actually happening.

This produced several corrections during live runs. Signals that looked neutral in aggregate revealed concentrated bleed in specific subclasses. The fix was not to tune the aggregate; it was to understand the subclass.

The analytical pipeline was built to support this naturally. Every live trade is attributed back to its originating signal through a series of joins — some clean, some requiring fuzzy matching on time, price, and direction when record identifiers were unreliable. Attribution coverage above 85% was the standard before any family-level conclusions were drawn.

A running concern throughout was echo: whether a structure that reappears in quick succession within the same zone carries different forward expectation than a first-occurrence structure. A lightweight tagging mechanism was added to the signal emission path to track this. Every analysis since then includes an explicit split on this tag — not because the answer is known, but because not asking the question would mean collecting data without ever reading it.

---

## The Temporal Hypothesis

There is a working hypothesis that the system's edge is concentrated in specific session windows where volatility and microstructure conditions align with the signal classes being traded. Outside those windows, the structural reads may still be directionally correct, but the execution environment — spread, range, follow-through — is such that even correct calls produce net-negative outcomes. Winners cap early; losers run.

This hypothesis was formed on the basis of a single session's data, which is not enough to act on. The current phase is collecting dense multi-day data precisely to test it. If confirmed, the response is a session-hours gate — a time-of-day filter applied at the candidate level — not a retune of the underlying detection parameters.

The broader architectural vision is similar: a time-invariant edge layer that detects and scores structures, paired with a time-bucketed execution layer that resolves the right management parameters for the current regime. The edge does not change with the time of day; the optimal way to manage a trade that the edge has identified may change substantially. Separating these concerns means the edge layer can be evaluated and evolved independently of the execution layer.

---

## Iteration Style

Progress was never linear. Each session produced findings that partially invalidated previous assumptions and opened new questions. The posture throughout was to treat each run as a data collection event first and a trading event second — the goal of the live system during the early phase was not to make money but to generate a statistically sufficient sample under real conditions.

This required a specific kind of discipline: not cutting underperforming components prematurely, not over-interpreting single-day results, and not proposing parameter changes in response to noise. The working standard was to collect enough data that conclusions could be drawn with some confidence before touching configuration. The cost of that discipline was accepting short-term bleed as the price of an unbiased sample.

A related discipline: changes were always made to the source and propagated explicitly to the running environment. No silent assumptions about what was live. When a feature appeared to not be working, the first step was always to verify that the version running was actually the version that had been written — not to assume logic was wrong before confirming that the logic was present.

---

## What This Is Not

This document describes a methodology, not a system. The specific signal classes, the gate conditions, the scoring weights, the execution parameters, the variant configurations, and the accumulated findings from live runs are not described here. That is the work product. What is described here is the process by which that work product was developed and the reasoning behind the choices made along the way.

The process is, in some ways, more transferable than the product. Markets change; edges erode; parameters that worked in one regime fail in another. A process that is hypothesis-driven, data-dense, dimensionally rigorous, and disciplined about the difference between noise and signal is what allows a system to be rebuilt when it needs to be rebuilt — and to know when that moment has arrived.
