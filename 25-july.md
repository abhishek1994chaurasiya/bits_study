Absolutely — here is the content formatted as a Markdown document.

# Lecture Overview

**Lecture:** NLP Applications — Prof. Chetana Anoop Gavankar
**Recording duration:** ~1:59:56
**Primary focus:** **Named Entity Recognition (NER)** as an Information Extraction (IE) task, followed by different approaches for NER:

1. Information Extraction recap
2. Named Entity Recognition and its subtasks
3. NER challenges
4. Rule-based NER
5. Machine-learning-based NER
6. BIO/IOB/BILOU tagging
7. MEMM
8. CRF
9. Domain-specific NER and gazetteers
10. Bi-LSTM for NER
11. Bi-LSTM + CRF
12. Transformer/BERT-based NER
13. Language models vs Transformers vs LLMs
14. LLM-based NER
15. RAG-based NER
16. Combining rules + transformers + LLM + RAG + agents
17. Ensemble NER
18. spaCy implementation
19. Custom NER and fine-tuning BERT
20. Real-world applications and implementation considerations

**Important:** The transcript contains some speech-recognition errors such as "RAJ/RAJAT" where the lecturer is clearly discussing **RAG**, and "BI-LSTM" sometimes appears as "by LSTM". I have normalized terminology where the intended meaning is clear, but I have **not invented missing mathematical details**.

---

# Learning Objectives

By the end of this lecture, you should be able to:

* Explain what **Information Extraction (IE)** is.
* Explain why **NER is a sequence-learning problem**.
* Distinguish **finding an entity** from **classifying its entity type**.
* Explain BIO/IOB tagging with examples.
* Explain rule-based NER and its advantages/limitations.
* Explain ML-based NER and the need for:

  * labelled data
  * feature engineering
  * entity classes
* Explain why NER is difficult across different domains and languages.
* Understand how **MEMM** performs sequential tagging.
* Explain the problem with greedy MEMM inference and the use of **beam search**.
* Understand the conceptual difference between **MEMM and CRF**.
* Explain why **Bi-LSTM** helps NER.
* Explain why **Bi-LSTM + CRF** can improve tagging.
* Explain why **transformers/attention** are useful for NER.
* Distinguish:

  * language model
  * transformer architecture
  * LLM
* Explain how generic LLMs can be used for NER and why they may not be appropriate for production/domain-specific NER.
* Explain **RAG-based NER**.
* Understand why a combination of rules, dictionaries, transformers, RAG and LLMs may be preferable in real-world systems.
* Explain how spaCy can be used for NER and why custom fine-tuning may be necessary.

---

# 5-Minute Revision

## 1. What is Information Extraction?

IE extracts **structured information from unstructured text**.

The lecture describes three major components:

> **NER + Relation Extraction + Event Detection**

Example:

> "Bill Gates founded Microsoft."

NER:

* Bill Gates → PERSON
* Microsoft → ORGANIZATION

Relation:

* Bill Gates → founded → Microsoft

The extracted information can subsequently be stored in a **database, knowledge base, knowledge graph or vector data store**.

**Timestamp:** ~00:15:00–00:21:20

---

## 2. What is NER?

NER has two important subtasks:

### Step 1 — Find the entity

Identify which portion of text is an entity.

### Step 2 — Classify the entity

Determine whether it is:

* PERSON
* LOCATION
* ORGANIZATION
* DATE
* etc.

**Timestamp:** ~00:15:15–00:15:50 and ~00:27:42–00:29:20

---

## 3. Why is NER difficult?

Because:

* Capitalization is not sufficient.
* The same word can have different meanings.
* Entity boundaries can be ambiguous.
* Entities can contain multiple words.
* Different domains have different entity types.
* Different languages behave differently.
* Abbreviations exist.
* Code mixing exists.
* Web documents have noise and complicated layouts.

Example:

> **Washington**

could refer to a PERSON or LOCATION.

Example:

> **Birla Institute of Technology and Science, Pilani**

must be treated as one entity rather than several independent pieces.

**Timestamp:** ~00:28:00–00:33:00

---

## 4. BIO tagging

For:

> Ratan Tata

BIO representation:

| Token | Tag      |
| ----- | -------- |
| Ratan | B-PERSON |
| Tata  | I-PERSON |

`O` = Outside any entity.

* **B** = Beginning
* **I** = Inside
* **O** = Outside

BIO/IOB tagging is repeatedly emphasized as important even for modern transformer-based NER.

**Timestamp:** ~00:29:39–00:30:12 and ~00:39:09–00:40:27

---

## 5. Rule-based NER

Uses:

* linguistic rules
* regular expressions
* dictionaries

Example:

If text matches a particular date pattern → DATE.

If a currency symbol appears with a number → possibly MONEY.

Advantages:

* No training data
* Cheap
* Easy for small domains
* Highly interpretable

Disadvantages:

* Rules are domain-specific.
* Rules may be language-specific.
* Rules become difficult to maintain as complexity grows.

**Timestamp:** ~00:35:14–00:38:10

---

## 6. ML-based NER

Requires:

1. Training documents
2. Labelled entities
3. Entity classes
4. Features
5. Classification algorithm

Possible classifiers mentioned:

* SVM
* Naive Bayes
* Decision Tree
* AdaBoost
* Gradient Boosting

Features can include:

* current word
* previous word
* next word
* context
* POS tags
* capitalization
* word length
* punctuation

**Timestamp:** ~00:38:13–00:49:00

---

## 7. MEMM

**Maximum Entropy Markov Model**

It is described as a **discriminative sequence model**.

It predicts the label of the current word using features such as the current/previous context.

Problem:

> Greedy left-to-right selection can make an early wrong decision that affects subsequent labels.

Solution discussed:

> **Beam search**

instead of always selecting only the single highest-probability label.

**Timestamp:** ~00:50:20–00:56:20

---

## 8. CRF

**Conditional Random Fields**

The lecturer presents CRF as another sequence model related to MEMM.

Important conceptual difference:

* MEMM makes conditional predictions using features.
* CRF considers dependencies between output labels and the sequence.

The lecturer explicitly says that **mathematical details of CRF are not the focus** and that she does not intend to ask mathematical problems on CRF in this context.

**Timestamp:** ~01:06:10–01:07:42
**Exam-related lecturer statement:** ~01:02:15–01:02:44

---

## 9. Bi-LSTM

Bi-LSTM processes information in:

* left → right
* right → left

This gives both-side context.

This is particularly useful for ambiguous entities.

Example:

> Ratan Tata Foundation

Looking only left-to-right may make **Ratan Tata** appear to be a person.

Looking at **Foundation** from the right side helps determine that the phrase may refer to an organization.

**Timestamp:** ~01:12:05–01:15:56

---

## 10. Bi-LSTM + CRF

The lecturer explains that CRF is added after Bi-LSTM to consider the relationship between output tags.

Conceptually:

**Bi-LSTM → contextual representation → CRF → sequence of tags**

CRF checks whether predicted labels are compatible with preceding labels.

**Timestamp:** ~01:16:57–01:19:20

---

## 11. Transformer/BERT for NER

Transformer's major advantage is **attention**.

Attention helps capture **long-range dependencies**.

This can help with:

* NER
* relation extraction
* coreference resolution
* contextual disambiguation

The lecture particularly emphasizes that transformers can capture context better than earlier approaches.

**Timestamp:** ~01:19:49–01:23:16

---

## 12. Transformer ≠ LLM

This distinction is important.

### Transformer

A **neural network architecture**.

### Language model

A concept/task involving prediction of the next token(s) based on previous tokens.

### LLM

A large pretrained language model, commonly built using transformer architectures.

The lecturer specifically explains that language modelling can also be performed using **n-gram models**, so language modelling itself is not synonymous with transformers.

**Timestamp:** ~01:25:07–01:29:45

---

## 13. LLM-based NER

A pretrained LLM can potentially identify:

* person
* organization
* location
* etc.

But problems include:

* hallucination
* insufficient domain knowledge
* lack of guaranteed production reliability
* need for fine-tuning for specialized domains

**Timestamp:** ~01:30:00–01:31:20

---

## 14. RAG-based NER

Instead of asking a generic LLM to "know" the entities, maintain a domain-specific knowledge base/knowledge graph.

Example BITS scenario:

> Faculty → Course → Topic → Program

During retrieval, the system retrieves relevant domain information and uses it to identify entities.

This provides **grounding**.

**Timestamp:** ~01:31:16–01:36:55

---

## 15. Real-world NER architecture

The lecture strongly emphasizes:

> **Don't use an expensive LLM/agent for every simple problem.**

Possible combination:

**Rules/Regex → Transformer → LLM → RAG → Agent**

depending on the task.

Example:

For PAN/Aadhaar/account numbers:

> Regex may be enough.

For more complex entities:

> Transformer/NER model may be appropriate.

For validation:

> RAG can provide grounding.

For complex workflows:

> Agents may be used.

**Timestamp:** ~01:37:08–01:39:30

---

## 16. Custom NER

Generic spaCy models may not recognize organization-specific entities correctly.

Example from lecture:

The model identified **NLP** as an organization when the intended meaning was a **course name**.

Solution:

* custom labelled dataset
* custom NER
* fine-tune BERT
* domain-specific NER model

**Timestamp:** ~01:50:33–01:55:20

---

# Key Concepts

## Concept 1 — Information Extraction

### Definition

Information Extraction converts unstructured text into structured information.

### Lecture explanation

The lecturer describes IE as extracting important information such as:

* who did something
* what they did
* whom they did it to
* when they did it
* relationships between entities

**Lecture:** ~00:20:14–00:21:19 and ~00:25:07–00:25:34

### Intuitive explanation

Think of a paragraph as a messy document and IE as converting it into a structured table.

Text:

> "Abhishek joined ABC Technologies in Mumbai in 2024."

Structured representation:

| Entity           | Type         |
| ---------------- | ------------ |
| Abhishek         | PERSON       |
| ABC Technologies | ORGANIZATION |
| Mumbai           | LOCATION     |
| 2024             | DATE         |

Then relations can be extracted.

### Deep understanding

IE is broader than NER.

A useful mental model:

**Text → Entities → Relations → Events → Structured representation**

---

# Detailed Explanations

## Concept 2 — NER

### Definition

Named Entity Recognition identifies spans of text representing named entities and assigns an entity type to them.

### Lecture's decomposition

The lecturer explicitly divides NER into:

**Find → Classify**

1. Find the entity span.
2. Determine its type.

**Timestamp:** ~00:15:15–00:15:50

### Example

> "Microsoft was founded by Bill Gates."

Entities:

* Microsoft → ORGANIZATION
* Bill Gates → PERSON

### Important distinction

NER is not simply:

> "Find capitalized words."

Capitalization is only one possible clue.

The lecture explicitly discusses cases where:

* a capitalized word is not an entity
* an entity is not properly capitalized
* numbers can be entities

**Timestamp:** ~00:27:56–00:28:42

---

# Concept 3 — Entity Boundary Detection

This is an important NER problem.

Consider:

> Birla Institute of Technology and Science, Pilani

The system must determine the **complete span**.

It cannot simply assume:

> "Take three words."

because entity lengths vary.

**Timestamp:** ~00:31:52–00:32:44

### Deep understanding

NER therefore involves two dimensions:

**Where is the entity?**

and

**What type is it?**

This is why BIO tagging becomes useful.

---

# Concept 4 — Ambiguity

### Example from lecture

**Washington**

could be:

* PERSON
* LOCATION

Similarly:

**Tata**

could refer to:

* a person
* an organization
* a product/company context

**Timestamp:** ~00:31:04–00:31:43

### Why this matters

The word itself may not contain enough information.

You need **context**.

This leads directly to why:

* sequential models
* Bi-LSTM
* attention
* transformers

are useful.

---

# Concept 5 — Coreference Resolution

### Definition

Coreference resolution determines when different expressions refer to the same entity.

### Lecture example

> "Chetana is teaching Conversational AI in BITS Pilani. She is also teaching NLP Applications."

Here:

**She → Chetana**

and:

**WILP → BITS Pilani**

**Timestamp:** ~00:33:37–00:34:15

### Connection

This is related to NER because identifying an entity is not always enough.

You may also need to know:

> "Does this expression refer to an entity mentioned earlier?"

---

# Concept 6 — Event Detection

The lecturer describes an event as a special type of relation involving time.

Example:

> NLP Applications is conducted from 3:40 PM to 5:40 PM.

You need to identify:

* event
* date/time
* duration

**Timestamp:** ~00:34:31–00:34:56

### IE relationship

The lecture presents:

**NER + Relation Extraction + Event Detection**

as major components of IE.

---

# Concept 7 — Rule-Based NER

### Lecture approach

Rules can use:

* regular expressions
* linguistic rules
* dictionaries

Example:

A date pattern can be recognized using a regular expression.

Currency symbols can provide clues for MONEY entities.

**Timestamp:** ~00:35:14–00:36:55

### Advantages

The lecturer explicitly emphasizes:

* zero training
* low cost
* easy for small domains
* high interpretability/explainability

**Timestamp:** ~00:37:24–00:38:10

### Why explainability matters

The lecturer stresses that industry systems need to understand **why** a particular result was produced.

This is a conceptual point worth understanding rather than merely memorizing.

---

# Concept 8 — Machine-Learning NER

### Pipeline

The lecture's ML-based approach can be understood as:

**Collect corpus**

↓

**Label entities**

↓

**Create features**

↓

**Train classifier**

↓

**Predict entity labels**

The lecturer describes these approaches as:

> corpus-based / training-data-based / data-hungry

**Timestamp:** ~00:38:23–00:39:02

### Features

Mentioned examples include:

* current word
* previous word
* next word
* context
* POS tags
* capitalization
* word length
* punctuation

**Timestamp:** ~00:43:49–00:45:58

### Important limitation

If your training data only contains certain entity classes, the model is limited by those labels.

Adding a new domain/entity type may require:

* new labelled data
* new features
* retraining

**Timestamp:** ~00:46:08–00:48:54

---

# Concept 9 — BIO/IOB Tagging

## BIO

| Tag | Meaning             |
| --- | ------------------- |
| B   | Beginning of entity |
| I   | Inside entity       |
| O   | Outside entity      |

Example:

> Barack Obama visited New York.

| Token   | Tag   |
| ------- | ----- |
| Barack  | B-PER |
| Obama   | I-PER |
| visited | O     |
| New     | B-LOC |
| York    | I-LOC |

### Why not just I/O?

Suppose:

> New York

Both words are part of LOCATION.

Without B, it becomes difficult to determine where one entity begins.

The lecturer explicitly explains this reason.

**Timestamp:** ~00:39:54–00:40:27

---

# Concept 10 — BILOU

The lecturer later mentions another tagging variant:

**BILOU**

The purpose is to provide more precise information about entity boundaries, including the ending/single-token case.

**Timestamp:** ~01:10:01–01:10:50

### Important

The lecturer calls it a **variant/optimization** of tagging rather than presenting it as a fundamentally different NER algorithm.

---

# Concept 11 — MEMM

### Full form

**Maximum Entropy Markov Model**

### Lecture classification

MEMM is presented as a **discriminative classifier/sequence model**.

**Timestamp:** ~00:50:20–00:51:15

### Basic idea

For each token:

> Consider features → calculate probabilities → choose a label.

Features can include:

* current word
* previous word
* context

**Timestamp:** ~00:51:34–00:51:47

---

## MEMM example

Consider:

> Ratan Tata Foundation

The model goes from left to right.

It may initially predict:

> Ratan → B-PER

Then:

> Tata → I-PER

But when it reaches:

> Foundation

the earlier decision may create problems.

The lecturer uses this example to explain why greedy sequential selection can produce incorrect entity boundaries.

**Timestamp:** ~00:52:33–00:54:36

---

# Concept 12 — Greedy Inference vs Beam Search

### Greedy approach

At each position:

> Select the currently highest-probability label.

Problem:

An early wrong decision can affect the final sequence.

### Beam Search

Instead of retaining only one possibility, maintain multiple high-probability sequences.

The lecturer specifically says:

> "We use the beam search."

and explains selecting the top candidates rather than only one.

**Timestamp:** ~00:55:32–00:56:16

### Exam importance

This is a good conceptual question:

> **Why can greedy decoding be problematic in sequence labelling, and how does beam search help?**

---

# Concept 13 — CRF

### Full form

**Conditional Random Field**

### Lecture explanation

CRF is another sequence model related to MEMM.

It predicts output conditionally based on:

* input/features
* output-label dependencies

The lecturer describes the probability formulation as conceptually related to softmax/conditional probability.

**Timestamp:** ~01:06:10–01:07:18

### Key conceptual difference

The lecturer emphasizes that CRF considers previous output/tag information, whereas MEMM's behaviour was described in contrast to this.

**Timestamp:** ~01:07:23–01:07:42

### Important exam note

The lecturer explicitly stated:

> she was not going into the mathematical details of CRF.

and indicated that she did not intend to ask mathematical CRF problems in this context.

**Timestamp:** ~01:02:15–01:02:44

**My exam assessment:** Understand the **conceptual difference**, not the derivation, unless later lectures change this expectation.

---

# Concept 14 — Gazetteer / Dictionary-Based NER

### Definition

A gazetteer is essentially a domain-specific dictionary/list of known entities.

The lecturer explicitly says:

> "Gadget here is nothing but dictionary."

**Timestamp:** ~01:08:30–01:08:59

### Example

For a medical domain, a dictionary might contain:

* disease names
* genes
* drugs
* procedures

If a word appears in the dictionary, it can provide evidence that it is an entity.

### Why useful?

Some domain-specific entities may be difficult for generic ML classifiers to recognize.

---

# Concept 15 — Domain-Specific NER

NER is not universal.

A model designed for:

**General English**

may not work well for:

**Medical text**

or:

**Legal text**

or:

**BITS-specific educational text**

The lecturer explicitly mentions specialized NER systems for:

* legal
* medical
* other domains

**Timestamp:** ~01:07:46–01:08:30

### Deep understanding

The important principle is:

> **Entity definitions and useful features depend on the domain.**

---

# Concept 16 — Bi-LSTM

### Why bidirectional?

A normal LSTM can process:

> left → right

A Bi-LSTM considers:

> left → right
> right → left

This provides context from both sides.

**Timestamp:** ~01:12:05–01:15:56

### Lecture example

> Ratan Tata Foundation

The word **Foundation** appears later.

A left-to-right-only model may have ambiguity around:

> Ratan Tata

The right-side context helps determine whether the phrase is a person or organization.

### Deep understanding

This is an important transition:

**NER → context → sequence models → bidirectional context**

---

# Concept 17 — Bi-LSTM + CRF

Architecture:

**Input**

↓

**Bi-LSTM**

↓

**CRF**

↓

**NER tags**

The Bi-LSTM provides contextual representations.

The CRF considers relationships between output tags.

**Timestamp:** ~01:16:57–01:19:20

### Why can this help?

Suppose certain tags normally follow other tags.

CRF can exploit those dependencies.

This is particularly useful for ensuring more coherent sequences of BIO tags.

---

# Concept 18 — Transformer-Based NER

### Main advantage

**Attention mechanism**

The lecturer contrasts this with Bi-LSTM, where she says she is not using attention scores.

**Timestamp:** ~01:19:31–01:20:44

### Why attention matters for NER

It helps capture:

> **long-range dependencies**

The lecturer connects this to:

* NER
* coreference
* relation extraction
* contextual understanding

**Timestamp:** ~01:20:44–01:23:16

### Deep understanding

The evolution is:

**Rule-based**

→ explicit rules

**ML**

→ handcrafted features + labelled data

**Bi-LSTM**

→ learned features + bidirectional context

**Bi-LSTM + CRF**

→ contextual representation + label dependency

**Transformer**

→ attention + broader contextual relationships

---

# Concept 19 — Language Model vs Transformer vs LLM

This distinction is particularly important because students often mix these terms.

## Transformer

**Architecture.**

Comparable at the conceptual level to architectures such as LSTM.

## Language Modelling

**Task/concept.**

Predict the next token(s) based on previous tokens.

## LLM

A large pretrained language model, commonly using transformer architectures.

### Important lecturer clarification

Language modelling does **not require transformers**.

It can also be implemented using:

> n-gram language models.

**Timestamp:** ~01:26:47–01:29:45

### Mental model

Think:

> **Transformer = architecture**

> **Language modelling = task**

> **LLM = large pretrained model**

This is one of the concepts I recommend understanding deeply rather than memorizing.

---

# Concept 20 — LLM-Based NER

A pretrained language model can potentially perform NER without training a new traditional classifier.

The lecturer explains that pretrained models have learned from large corpora and can identify entities.

**Timestamp:** ~01:30:00–01:30:32

### But why not always use an LLM?

Problems:

* hallucination
* domain-specific terminology
* production reliability
* need for fine-tuning

**Timestamp:** ~01:30:32–01:31:20

---

# Concept 21 — RAG-Based NER

This is one of the more modern and interesting parts of the lecture.

## Basic idea

Suppose you have a BITS-specific application.

You maintain a knowledge base/knowledge graph containing:

* faculty
* courses
* topics
* programs
* relationships

Then the system can retrieve relevant information when identifying entities.

**Timestamp:** ~01:31:16–01:32:56

### Example

Knowledge graph:

```text
Chetana
   |
 teaches
   |
NLP Applications
   |
belongs to
   |
BITS Pilani
```

If a user's text mentions:

> "Who teaches NLP Applications?"

retrieval can ground the answer using the stored domain knowledge.

### Why RAG helps

The generic LLM may not know:

> "Who is teaching this particular course this semester?"

But a domain-specific knowledge base can contain that information.

---

# Concept 22 — Hybrid NER Architecture

This is arguably one of the most important practical messages from the lecture.

The lecturer says you should **not automatically use the most powerful/expensive technology for every task**.

For example:

### Simple structured entities

PAN number / Aadhaar number / invoice number

→ Regex

### General entities

→ Transformer-based NER

### Domain-specific validation

→ RAG/knowledge base

### Complex workflow

→ Agent

**Timestamp:** ~01:37:08–01:39:30

### Deep understanding

Technology selection should depend on:

> **complexity + accuracy requirement + cost + explainability + domain**

rather than:

> "LLM is newer, therefore use LLM."

---

# Concept 23 — Ensemble NER

The lecturer also discusses combining multiple NER approaches.

For example:

**Dictionary NER**

*

**CRF**

*

**Transformer**

*

other approaches

Then combine/validate the outputs.

**Timestamp:** ~01:39:41–01:40:06

### Why?

Different methods have different strengths.

A dictionary may be excellent for known domain entities.

A transformer may be better for contextual entities.

A rule may be excellent for structured patterns.

---

# Concept 24 — spaCy NER

The lecturer demonstrates a simple NER implementation using **spaCy**.

She notes that it is:

* simple
* fast
* easy to implement

**Timestamp:** ~01:50:05–01:50:25

### Important practical limitation

The pretrained model may not understand your organization's custom entities.

The lecture demonstrates an example where the intended entity:

> NLP → course name

was classified incorrectly as:

> ORGANIZATION

**Timestamp:** ~01:52:38–01:53:09

### Lesson

A pretrained model is not automatically suitable for every domain.

---

# Concept 25 — Fine-Tuning BERT for Custom NER

The lecturer proposes:

> Fine-tune an existing BERT model using your own labelled dataset.

**Timestamp:** ~01:53:40–01:55:20

### Required ingredients

* custom dataset
* BIO-encoded labels
* appropriate entity classes
* model fine-tuning
* hyperparameter tuning
* evaluation

The lecturer also notes that fine-tuning requires computational resources/GPU.

**Timestamp:** ~01:53:45–01:55:09

---

# Important Definitions

| Term                       | Definition                                                               |
| -------------------------- | ------------------------------------------------------------------------ |
| **Information Extraction** | Extracting structured information from unstructured text                 |
| **NER**                    | Finding entity spans and assigning entity types                          |
| **Named Entity**           | A meaningful entity in text such as person, organization, location, date |
| **Relation Extraction**    | Identifying relationships between entities                               |
| **Event Detection**        | Detecting events and associated temporal information                     |
| **Coreference Resolution** | Determining when different expressions refer to the same entity          |
| **BIO**                    | Beginning-Inside-Outside entity tagging                                  |
| **BILOU**                  | A more detailed entity-boundary tagging scheme                           |
| **MEMM**                   | Maximum Entropy Markov Model                                             |
| **CRF**                    | Conditional Random Field                                                 |
| **Gazetteer**              | Dictionary/list of known entities                                        |
| **Bi-LSTM**                | Bidirectional Long Short-Term Memory network                             |
| **Transformer**            | Neural network architecture using attention                              |
| **Language Model**         | Model/task concerned with predicting tokens in language                  |
| **LLM**                    | Large pretrained language model                                          |
| **RAG**                    | Retrieval-Augmented Generation                                           |
| **Fine-tuning**            | Further training an existing model on task/domain-specific data          |
| **Ensemble NER**           | Combining multiple NER approaches/models                                 |

---

# Formulas / Rules

## 1. BIO Rules

For a multi-token PERSON:

```text
Ratan → B-PER
Tata  → I-PER
```

For a non-entity:

```text
visited → O
```

### Rule

`B` starts an entity.

`I` continues the same entity.

`O` means outside an entity.

---

## 2. HMM vs MEMM

The lecturer revisits an earlier NLP concept:

### HMM

Presented as:

> **generative**

Uses probability concepts involving prior and likelihood.

**Timestamp:** ~00:51:08–00:51:27

### MEMM

Presented as:

> **discriminative**

Uses conditional prediction/classification.

**Timestamp:** ~00:51:08–00:51:47

---

## 3. Conditional Probability Concept in CRF

The lecturer describes CRF conceptually as:

> predicting the output conditionally on features/input.

**Timestamp:** ~01:06:25–01:07:18

The exact mathematical formulation is **not clearly given in the transcript**, so I would not memorize an invented equation from this lecture.

---

# Examples

## Example 1 — Basic NER

Sentence:

> "Barack Obama visited the United States."

Expected:

```text
Barack Obama → PERSON
United States → LOCATION
```

The lecturer uses essentially this type of example when discussing Bi-LSTM.

**Timestamp:** ~01:12:43–01:13:10

---

## Example 2 — Ambiguous entity

> Washington

Could be:

```text
PERSON
```

or:

```text
LOCATION
```

Context is needed.

**Timestamp:** ~00:31:04–00:31:22

---

## Example 3 — Entity boundary

> Birla Institute of Technology and Science, Pilani

The system should ideally identify the **complete span** rather than individual fragments.

**Timestamp:** ~00:31:52–00:32:44

---

## Example 4 — Coreference

> Chetana is teaching NLP. She is also teaching Conversational AI.

`She` → `Chetana`

**Timestamp:** ~00:33:49–00:34:15

---

## Example 5 — Regex

Structured entity:

> Invoice number / PAN / Aadhaar / account number

A regular expression may be more appropriate than a large language model.

**Timestamp:** ~01:37:12–01:38:36

---

## Example 6 — Custom domain

Suppose your company has:

> "Customer 360"

as a specific entity type.

A generic NER model may not recognize it correctly.

Possible solution:

> labelled custom data → fine-tuned NER model.

**Lecture basis:** ~01:50:33–01:55:09

---

# Common Misconceptions

## Misconception 1: "NER means finding capitalized words."

**Wrong.**

Capitalization is only one possible feature.

The lecturer explicitly discusses:

* capitalized non-entities
* lowercase entities
* numbers as entities

**~00:27:56–00:28:42**

---

## Misconception 2: "NER only means identifying the entity."

**Wrong.**

There are two major subtasks:

> **Find + Classify**

**~00:15:15–00:15:50**

---

## Misconception 3: "LLM replaced all traditional NER."

**Wrong.**

The lecturer explicitly discusses continued use of:

* rules
* dictionaries
* ML
* CRF
* Bi-LSTM
* transformers
* domain-specific NER
* hybrid systems

**~00:35:14–00:39:00 and ~01:37:08–01:40:06**

---

## Misconception 4: "Transformer and LLM mean the same thing."

**Wrong.**

The lecturer distinguishes:

> Transformer = architecture

> LLM = pretrained language model

**~01:25:07–01:29:45**

---

## Misconception 5: "Language modelling requires transformers."

**Wrong.**

The lecturer explicitly mentions:

> n-gram language models.

**~01:26:47–01:29:45**

---

## Misconception 6: "A pretrained spaCy model will understand my organization's entities."

**Not necessarily.**

The lecturer demonstrates a custom-entity misclassification.

**~01:52:38–01:53:09**

---

## Misconception 7: "More powerful model = better engineering choice."

**Not necessarily.**

For simple structured entities, regex/dictionaries can be cheaper, faster and more interpretable.

**~01:37:08–01:39:30**

---

# Connections to Previous Concepts

This lecture deliberately builds on several earlier NLP concepts.

## 1. POS Tagging → NER

The lecturer repeatedly compares NER to POS tagging.

Both can be viewed as **sequence labelling problems**.

**~00:44:31–00:44:58**

---

## 2. HMM → MEMM → CRF

The sequence-model progression is:

```text
HMM
 ↓
MEMM
 ↓
CRF
```

The lecturer connects these to previously studied NLP material.

**~00:50:20–00:51:27**

---

## 3. LSTM → Bi-LSTM

Earlier sequence models lead naturally into:

> Bi-LSTM

because NER benefits from both left and right context.

**~01:12:05–01:15:56**

---

## 4. Bi-LSTM → Bi-LSTM + CRF

CRF adds output-label dependency modelling.

**~01:16:57–01:19:20**

---

## 5. Bi-LSTM → Transformer

The major transition is:

> sequential context → attention-based context

Transformers improve the ability to capture long-range dependencies.

**~01:19:49–01:23:16**

---

## 6. Transformer → LLM → RAG

The lecture then moves from:

```text
Transformer architecture
        ↓
Pretrained language models / LLMs
        ↓
Domain-specific problems
        ↓
RAG for grounding
```

**~01:25:07–01:31:20**

More precisely, the RAG discussion begins around **01:31:16**.

---

# Exam Focus

> **These are my assessments of likely exam value, not guarantees about the examination.** Where the lecturer explicitly discussed assessment, I identify that separately.

## 🔴 Very High Priority — Understand deeply

### 1. NER as Find + Classify

Know:

> Entity span detection + entity type classification

**~00:15:15–00:15:50**

---

### 2. BIO tagging

Be able to label a sentence.

**~00:39:09–00:40:27**

Likely question type:

> Given a sentence, generate BIO tags.

---

### 3. NER challenges

Especially:

* ambiguity
* entity boundaries
* domain variation
* language variation
* abbreviations

**~00:28:00–00:33:00**

---

### 4. Rule-based vs ML-based NER

Know:

| Rule-based                   | ML-based                  |
| ---------------------------- | ------------------------- |
| Rules/regex/dictionaries     | Training data             |
| Little/no training           | Labelled corpus           |
| Highly interpretable         | Learns patterns           |
| Good for constrained domains | More adaptable with data  |
| Maintenance can be difficult | Data/feature requirements |

---

### 5. MEMM + greedy vs beam search

Understand **why greedy decoding can fail**.

**~00:52:33–00:56:16**

---

### 6. MEMM vs CRF

Understand conceptually.

**~01:06:10–01:07:42**

The lecturer explicitly said not to focus on CRF mathematics in this lecture.

---

### 7. Bi-LSTM for NER

Know why both directions are useful.

**~01:12:05–01:15:56**

---

### 8. Bi-LSTM + CRF

Understand why CRF improves sequence consistency.

**~01:16:57–01:19:20**

---

### 9. Why Transformers help NER

Core answer:

> **Attention → better contextual/long-range dependency handling**

**~01:19:49–01:23:16**

---

### 10. Transformer vs Language Model vs LLM

This is an excellent conceptual exam question.

**~01:25:07–01:29:45**

---

### 11. RAG-based NER

Understand why domain knowledge can improve NER.

**~01:31:16–01:36:55**

---

### 12. Hybrid NER

Understand why you might combine:

> Regex + dictionary + transformer + LLM + RAG + agent

rather than blindly using an LLM.

**~01:37:08–01:39:30**

---

# Lecturer's Explicit Assessment Signals

A few comments are particularly useful for exam preparation:

### CRF mathematics

At approximately **01:02:15**, the lecturer says she is not going into the mathematical details and does not intend to ask mathematical problems on CRF in this context.

### Neural/Transformer mathematics

At approximately **01:11:35–01:12:05**, she says she is focusing on **application/use of the algorithms for NER**, rather than asking students to mathematically compute forward/backpropagation.

### NER concepts

At approximately **01:02:24–01:02:44**, she emphasizes that students should **note the concepts**.

### Implementation

At approximately **01:46:34–01:46:58**, she says implementation assignments may cover things such as:

* NER
* relation extraction
* temporal event extraction
* statistical machine translation
* privacy/ethics/guardrails

This is a lecturer statement about possible assignment/application work, not a guarantee of exam questions.

---

# Questions I Should Be Able to Answer

## Basic

1. What is Information Extraction?
2. What are the major components of an IE pipeline?
3. What is NER?
4. What are the two main subtasks of NER?
5. What is a named entity?
6. What is BIO tagging?
7. What do B, I and O represent?
8. What is BILOU?
9. What is coreference resolution?
10. What is event detection?

## Conceptual

11. Why is NER considered a sequence-labelling problem?
12. Why can't capitalization alone solve NER?
13. Why is entity-boundary detection difficult?
14. Why is NER domain-dependent?
15. Why does ML-based NER require labelled data?
16. What features can be used for traditional ML-based NER?
17. Why is feature engineering required in traditional ML NER?
18. What is a gazetteer?
19. Why are dictionaries useful for NER?

## MEMM / CRF

20. What is MEMM?
21. Is MEMM generative or discriminative?
22. How does MEMM perform sequential tagging?
23. What is the problem with greedy MEMM inference?
24. How does beam search address this?
25. What is CRF?
26. How does CRF differ conceptually from MEMM?
27. Why can CRF improve sequence tagging?

## Deep Learning

28. Why is Bi-LSTM useful for NER?
29. Why is bidirectional context useful?
30. Why can Bi-LSTM outperform a one-directional model in ambiguous cases?
31. Why combine Bi-LSTM with CRF?
32. What does CRF add to Bi-LSTM?

## Transformers / LLM

33. What is attention?
34. Why is attention useful for NER?
35. What are long-range dependencies?
36. Why are transformers useful for NER?
37. What is the difference between a transformer and an LLM?
38. What is language modelling?
39. Can language modelling be performed without transformers?
40. Why can a generic LLM be problematic for domain-specific NER?

## RAG / Real World

41. What is RAG-based NER?
42. How can a knowledge graph help NER?
43. Why can RAG provide better domain grounding?
44. When should you use regex instead of an LLM?
45. Why might an organization combine multiple NER approaches?
46. What is ensemble NER?
47. Why might a pretrained spaCy model fail on custom entities?
48. How would you build a custom NER system?
49. What is fine-tuning?
50. Why is labelled BIO data needed when fine-tuning BERT for NER?

---

# Timestamp Index

| Timestamp    | Topic                                      |
| ------------ | ------------------------------------------ |
| **00:07:26** | Information Extraction recap               |
| **00:09:27** | Types of named entities                    |
| **00:14:15** | Basic NER problem                          |
| **00:15:15** | NER = find + classify                      |
| **00:16:17** | Entity association/relations               |
| **00:18:00** | Capitalization as NER clue                 |
| **00:18:09** | Language-specific issues                   |
| **00:19:15** | Cost/when not to use LLM                   |
| **00:20:14** | NER, relation extraction, event detection  |
| **00:21:19** | Overall IE pipeline                        |
| **00:21:29** | IE challenges                              |
| **00:23:17** | Code mixing/language/genre                 |
| **00:23:29** | Resume extraction/layout                   |
| **00:25:07** | Structured information extraction          |
| **00:27:42** | Beginning NER-specific discussion          |
| **00:27:56** | Finding entities                           |
| **00:28:42** | Classifying entity types                   |
| **00:29:39** | BIO labelling                              |
| **00:30:24** | NER challenges                             |
| **00:31:04** | Washington ambiguity                       |
| **00:31:52** | Multi-word entity boundaries               |
| **00:32:52** | Entity subtypes                            |
| **00:33:49** | Coreference resolution                     |
| **00:34:31** | Event detection                            |
| **00:35:14** | Rule-based NER                             |
| **00:36:04** | Regex/rules                                |
| **00:37:24** | Rule-based advantages                      |
| **00:38:23** | ML-based NER                               |
| **00:39:09** | Labelled data + BIO                        |
| **00:39:54** | BIO: B/I/O meaning                         |
| **00:41:01** | Feature engineering                        |
| **00:42:06** | ML classifiers for NER                     |
| **00:43:49** | NER features                               |
| **00:44:31** | NER as sequence learning                   |
| **00:45:26** | Capitalization/length/punctuation features |
| **00:46:08** | ML limitations                             |
| **00:48:21** | Domain-specific NER challenges             |
| **00:49:22** | Knowledge graphs/gazetteers                |
| **00:50:20** | MEMM                                       |
| **00:51:08** | MEMM vs HMM                                |
| **00:52:33** | MEMM sequential tagging                    |
| **00:54:07** | Greedy inference problem                   |
| **00:55:32** | Greedy vs beam search                      |
| **01:02:15** | CRF mathematical/exam discussion           |
| **01:06:10** | CRF introduction                           |
| **01:07:23** | MEMM vs CRF                                |
| **01:08:07** | Domain-specific NER                        |
| **01:08:30** | Gazetteer/dictionary                       |
| **01:09:14** | BIO tagging across models                  |
| **01:10:01** | BILOU                                      |
| **01:11:11** | Neural approaches to NER                   |
| **01:12:05** | Bi-LSTM                                    |
| **01:12:32** | BERT/encoder-only for NER                  |
| **01:14:12** | Bi-LSTM without manual features            |
| **01:14:30** | Ratan Tata Foundation example              |
| **01:15:13** | Why bidirectional?                         |
| **01:16:57** | Bi-LSTM + CRF                              |
| **01:17:33** | CRF output dependencies                    |
| **01:19:49** | Transformer attention                      |
| **01:20:44** | Why transformers for NER                   |
| **01:21:17** | Long-range dependencies                    |
| **01:22:06** | Context/coreference/relations              |
| **01:23:05** | Transformers for NER                       |
| **01:25:07** | Transformer architecture                   |
| **01:25:42** | GPT/Claude as language models              |
| **01:26:47** | Language modelling                         |
| **01:27:09** | n-gram vs transformer language models      |
| **01:29:35** | Transformer vs language model              |
| **01:30:00** | LLM-based NER                              |
| **01:30:32** | LLM limitations/hallucination              |
| **01:31:16** | RAG-based NER                              |
| **01:31:29** | Knowledge graph example                    |
| **01:32:36** | Medical/legal domain RAG                   |
| **01:33:01** | Agentic NER / overkill                     |
| **01:34:22** | Concrete BITS RAG example                  |
| **01:35:50** | RAG identifying entity type                |
| **01:37:08** | Regex-based real-world NER                 |
| **01:38:27** | Hybrid NER workflow                        |
| **01:39:41** | Ensemble NER                               |
| **01:46:34** | Possible implementation areas              |
| **01:48:35** | spaCy NER output/entity positions          |
| **01:50:05** | spaCy implementation                       |
| **01:50:33** | Custom entities                            |
| **01:52:38** | spaCy misclassification example            |
| **01:53:40** | Fine-tuning BERT                           |
| **01:54:21** | BIO data for fine-tuning                   |
| **01:54:47** | Hyperparameter tuning                      |
| **01:55:20** | Domain-specific NER models/datasets        |
| **01:56:54** | Final CRF clarification                    |

---

# The Big Picture You Should Remember

If you remember only one conceptual flow from this lecture, make it this:

```text
                    INFORMATION EXTRACTION
                             │
             ┌───────────────┼────────────────┐
             ↓               ↓                ↓
            NER         Relation Extraction   Events
             │
       Find + Classify
             │
     ┌───────┴────────┐
     ↓                ↓
 Rule-based          ML-based
 Regex/Dictionary    Features + Labels
     │                │
     └───────┬────────┘
             ↓
            MEMM
             ↓
            CRF
             ↓
          Bi-LSTM
             ↓
       Bi-LSTM + CRF
             ↓
        Transformer
             ↓
       BERT / NER Models
             ↓
            LLM
             ↓
       RAG / Knowledge Graph
             ↓
     Hybrid / Ensemble NER
```

**The deepest idea is not memorizing this hierarchy.** Understand *why* each successive approach is useful:

> **Rules → explicit knowledge**
> **ML → learned patterns**
> **Bi-LSTM → bidirectional context**
> **CRF → label-sequence dependencies**
> **Transformer → attention and long-range context**
> **LLM → large pretrained language knowledge**
> **RAG → external/domain-specific grounding**
> **Hybrid systems → choose the appropriate tool for each subproblem**

That conceptual chain ties almost the entire two-hour lecture together.
