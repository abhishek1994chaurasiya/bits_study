# NLP Applications — Relation Extraction
## University Exam-Preparation Study Notes

**Lecture:** Relation Extraction  
**Lecturer:** Prof. Chetana Anoop Gavankar  
**Date:** August 1, 2026  
**Recording duration:** 2h 1m 36s

---

# Lecture Overview

This lecture focuses on **Relation Extraction (RE)** as part of Information Extraction (IE). It builds directly on Named Entity Recognition (NER) from the previous lecture.

```text
Information Extraction
        │
        ├── Named Entity Recognition (previous lecture)
        │
        ├── Relation Extraction  ← THIS LECTURE
        │
        └── Temporal/Event Extraction ← next session
```

Major topics:

1. Information Extraction recap
2. Relation Extraction
3. NER → Relation Extraction
4. Relation types and direction
5. RE as classification
6. Rule-based RE
7. Supervised ML
8. Feature engineering
9. Deep learning
10. Transformers and attention
11. LLM-based RE
12. Few-shot / in-context learning
13. Knowledge Graphs and RDF triples
14. Bootstrapping
15. Distant Supervision
16. Open Information Extraction
17. Domain-specific RE
18. Multilingual RE
19. Production considerations
20. Hybrid approaches

---

# Learning Objectives

You should be able to:

- Define Relation Extraction.
- Explain RE versus NER.
- Explain why NER is useful before RE.
- Identify whether two entities have a relation.
- Classify the relation type and direction.
- Represent relations as tuples/RDF triples.
- Explain rule-based and supervised RE.
- Explain traditional RE features.
- Explain feature engineering versus learned representations.
- Explain transformer-based RE.
- Explain bootstrapping and distant supervision.
- Distinguish bootstrapping from distant supervision.
- Explain Open IE.
- Explain LLM-based/few-shot RE.
- Discuss LLM production trade-offs.
- Explain domain-specific and multilingual RE.
- Explain why hybrid approaches are useful.

---

# 5-Minute Revision

## Relation Extraction

RE determines whether two entities are related and, if so, what relation exists.

Example:

```text
Tim Cook → PERSON
Apple    → ORGANIZATION

Tim Cook --CEO_OF--> Apple
```

## NER → RE

```text
Text
 ↓
NER
 ↓
Entities + types
 ↓
Relation Extraction
 ↓
Relation + type + direction
```

## Relation Extraction as classification

For an entity pair, predict a relation such as:

```text
CURE
PREVENT
CAUSE
```

## Rule-based RE

Uses:

- regular expressions
- linguistic patterns
- dependency parsing
- dictionaries/gazetteers

Strength: potentially high precision.

Weakness: limited recall and rule-maintenance burden.

## Supervised RE

```text
Labelled examples
      ↓
Features
      ↓
Classifier
      ↓
Relation
```

## Traditional features

- words between entities
- POS tags
- dependency relations
- head words
- n-grams
- entity types
- context

## Deep learning

Traditional ML relies more on manually engineered features. Deep learning learns representations/features automatically.

## Transformers

Transformers provide contextual representations using attention and can capture semantic similarity across different expressions.

## Bootstrapping

```text
Seed examples → more examples
```

A data-generation technique.

## Distant supervision

```text
Knowledge Graph + Corpus
          ↓
Training examples
```

## Knowledge Graph

```text
Subject — Predicate — Object
```

Example:

```text
Steve Jobs — FOUNDED — Apple
```

## LLM-based RE

LLMs can extract relations through prompts and few-shot/in-context examples.

Production concerns include:

- cost
- latency
- reproducibility
- domain suitability

## Industry lesson

> Start simple and use the simplest approach that solves the problem adequately.

---

# Key Concepts

| Concept | Exam Priority |
|---|---|
| Relation Extraction | Very High |
| NER → RE | Very High |
| Relation classification | Very High |
| Relation direction | Very High |
| Rule-based RE | Very High |
| Supervised RE | Very High |
| Feature engineering | Very High |
| Bootstrapping | Very High |
| Distant supervision | Very High |
| Knowledge Graph ↔ RE | Very High |
| Transformers/attention | High |
| LLM-based RE | High |
| Production trade-offs | High |
| Domain-specific RE | High |
| Multilingual RE | Medium |
| Open IE | Medium |

---

# Detailed Explanations

## 1. Information Extraction

### Definition

Information Extraction extracts structured information from unstructured text.

The lecture connects IE with tasks such as:

```text
NER
Relation Extraction
Event/Temporal Extraction
Coreference Resolution
```

### Intuitive explanation

Given:

> Steve Jobs founded Apple in California in 1976.

An IE system can produce:

```text
Steve Jobs → PERSON
Apple → ORGANIZATION
California → LOCATION
1976 → DATE

Steve Jobs --FOUNDED--> Apple
Apple --LOCATED_IN--> California
```

### Technical explanation

IE converts free-form language into structured entities, relations and other information that can be stored or processed computationally.

### Exam relevance

Understand IE as the larger umbrella under which RE belongs.

---

## 2. Relation Extraction

### Definition

Relation Extraction identifies semantic relationships between entities in text.

Conceptually:

```text
R(e1, e2)
```

where `e1` and `e2` are entities and `R` is a relation such as:

```text
FOUNDED
WORKS_FOR
LOCATED_IN
CAUSES
CURES
PREVENTS
```

### Intuitive explanation

NER asks:

> What are the important things/entities?

RE asks:

> How are those entities connected?

### Example

> Steve Jobs founded Apple.

```text
Steve Jobs --FOUNDED--> Apple
```

### Common misconception

Two entities appearing in the same sentence do not necessarily have a meaningful relation.

### Timestamp

**~00:02:10 onward** — Introduction to Relation Extraction and comparison with NER.

---

## 3. Relation Type and Direction

### Definition

A relation has a semantic type and often a direction from one entity to another.

Example:

```text
Paris --LOCATED_IN--> France
```

not:

```text
France --LOCATED_IN--> Paris
```

### Why it matters

The same pair of entities can potentially have several relations:

```text
WORKS_FOR
FOUNDED
CEO_OF
OWNS
INVESTOR_IN
```

### Exam relevance

Be prepared to explain why identifying the correct direction matters.

### Timestamp

**~00:10:00–00:20:00** — Relation types, entity pairs and extraction challenges.

---

## 4. NER → Relation Extraction

### Definition

NER identifies entities and their types; RE identifies relations between those entities.

### Pipeline

```text
Text
 ↓
NER
 ↓
Ratan Tata → PERSON
Tata Sons → ORGANIZATION
 ↓
RE
 ↓
Ratan Tata --FOUNDED--> Tata Sons
```

### Why NER helps

Knowing entity types reduces the possible relation space.

For example:

```text
PERSON + ORGANIZATION
```

makes relations such as `WORKS_FOR` or `FOUNDED` plausible.

### Deep understanding

This is one of the most important lecture connections:

```text
WHO/WHAT?
    ↓
NER

HOW CONNECTED?
    ↓
RE
```

### Timestamp

**~00:02:10–00:10:00**

---

## 5. Relation Extraction as Classification

### Definition

For an entity pair and its context, a classifier can predict the relation class.

Example:

```text
Entity pair:
Drug X, Disease Y

Possible classes:
CURE
PREVENT
CAUSE
```

### Conceptual formulation

```text
Input:
(e1, e2, context)

Output:
relation ∈ {R1, R2, ..., Rn}
```

### Exam relevance

Very important. Be able to explain RE as a classification problem.

### Timestamp

**00:23:58–00:25:00** and surrounding supervised-learning discussion.

---

## 6. Rule-Based Relation Extraction

### Definition

A rule-based RE system manually defines patterns that correspond to relations.

Example:

```text
X works at Y
→ X --WORKS_FOR--> Y
```

Another:

```text
X is located in Y
→ X --LOCATED_IN--> Y
```

### Techniques

- Regular expressions
- Dependency parsing
- Linguistic patterns
- Dictionaries/gazetteers

### Advantages

- High precision for well-designed patterns
- Explainable
- Low training-data requirement
- Good for controlled domains

### Disadvantages

- Low recall when patterns are not covered
- Rules require maintenance
- Difficult to scale to many relations/domains

### Common misconception

Rule-based RE is not "obsolete." It remains useful when the domain and patterns are controlled.

### Timestamp

**~00:20:00–00:30:00**

---

## 7. Supervised Relation Extraction

### Definition

A supervised model learns relation patterns from labelled examples.

### Pipeline

```text
Corpus
 ↓
Entity identification
 ↓
Relation labels
 ↓
Training examples
 ↓
Feature extraction
 ↓
Classifier
 ↓
Prediction
```

### Example

| Entity 1 | Entity 2 | Relation |
|---|---|---|
| Drug A | Disease X | CURES |
| Drug B | Disease Y | PREVENTS |
| Virus X | Disease Z | CAUSES |

### Main limitation

High-quality labelled data is expensive.

### Timestamp

**~00:23:16 onward** and **~00:30:00–00:50:00**.

---

## 8. Feature Engineering

### Definition

Feature engineering means manually constructing useful signals for a machine-learning model.

### Features mentioned/discussed

- words between entities
- POS tags
- dependency parsing
- head word
- n-grams
- entity types
- contextual words
- left/right context

### Example

> Barack Obama was born in Hawaii.

Entities:

```text
Barack Obama → PERSON
Hawaii → LOCATION
```

Useful features:

```text
Entity 1 type = PERSON
Entity 2 type = LOCATION
Words between = "born in"
Dependency information
```

These can strongly indicate:

```text
BORN_IN
```

### Common misconception

Feature engineering is not just "choosing words." It can include syntactic, positional and entity-type information.

### Timestamp

**~00:35:00–00:45:00** and **01:12:16–01:14:27**.

---

## 9. Traditional ML vs Deep Learning

### Traditional ML

```text
Text
 ↓
Human-designed features
 ↓
Classifier
 ↓
Relation
```

### Deep Learning

```text
Text
 ↓
Embedding
 ↓
Neural network
 ↓
Learned representation
 ↓
Relation
```

### Key distinction

Traditional ML generally requires explicit feature engineering.

Deep learning learns useful representations/features automatically.

### Deep understanding

Do not memorize "deep learning has no features." It has learned representations; the important distinction is how they are obtained.

---

## 10. Dependency Parsing

### Definition

Dependency parsing represents syntactic relationships between words.

### Why useful for RE

The syntactic path between two entities can provide a strong clue about their semantic relationship.

Example:

> John works for Microsoft.

The dependency structure around `works` and `for` can help identify:

```text
John --WORKS_FOR--> Microsoft
```

### Exam relevance

Know dependency parsing as a traditional RE feature.

---

## 11. Gazetteers

### Definition

A gazetteer is a dictionary/list of known entities or terms.

Example:

```text
Countries:
India
USA
France
Japan
```

or:

```text
Diseases:
Cancer
Diabetes
Malaria
```

### Why useful?

If the possible entities are already known, a gazetteer can simplify entity identification and constrain possible relations.

---

## 12. Knowledge Graphs and RDF Triples

### Definition

An RDF-style triple represents:

```text
Subject — Predicate — Object
```

Example:

```text
Steve Jobs — FOUNDED — Apple
```

Equivalent tuple:

```text
(Steve Jobs, FOUNDED, Apple)
```

### RE → Knowledge Graph

```text
Text
 ↓
NER
 ↓
RE
 ↓
Triples
 ↓
Knowledge Graph
```

### Knowledge Graph → RE

Known relations in a knowledge graph can also help create or identify extraction examples.

### Exam relevance

Very high. Understand the two-way relationship.

---

## 13. Bootstrapping

### Definition

Bootstrapping starts with a small set of seed examples and uses them to generate additional examples.

Conceptually:

```text
Seed examples
      ↓
Pattern discovery
      ↓
More examples
      ↓
Larger dataset
```

### Example

Start with:

```text
"Einstein was born in Ulm."
```

Possible pattern:

```text
PERSON + "born in" + LOCATION
```

Find other sentences matching related patterns.

### Important

The lecture frames bootstrapping as a **data-generation technique**, not simply as the final classifier.

### Timestamp

**~00:55:00–01:01:33**

---

## 14. Distant Supervision

### Definition

Distant supervision uses known relations from a knowledge graph/knowledge base together with a corpus to automatically produce training examples.

### Example

Knowledge graph:

```text
Albert Einstein --BORN_IN--> Ulm
```

Corpus:

> Albert Einstein was born in Ulm.

The sentence can be associated with the `BORN_IN` relation.

### Why useful?

Manual annotation is expensive, so automatic generation can create large training sets.

### Timestamp

**01:03:03 onward**, especially **~01:03:00–01:10:00**.

---

## 15. Bootstrapping vs Distant Supervision

This distinction is **very exam-relevant**.

| Bootstrapping | Distant Supervision |
|---|---|
| Starts with seed examples | Starts with known KG relations |
| Generates more examples | Uses KG + corpus |
| Pattern/data-generation approach | Automatically creates training examples |
| Can begin with very few examples | Depends on a knowledge source |

### Memory trick

```text
Bootstrapping:
small examples → more examples

Distant supervision:
KG + corpus → training examples
```

---

## 16. Transformer-Based RE

### Definition

Transformers are deep-learning architectures using attention to build contextual representations.

### Why useful for RE

Traditional rules may distinguish exact phrases:

```text
born in
birth of
```

Transformers can learn semantic relationships between differently worded expressions.

### Conceptual flow

```text
Text
 ↓
Token representations
 ↓
Attention
 ↓
Contextual representations
 ↓
Relation prediction
```

### Timestamp

**01:01:33–01:03:03** and later transformer/deep-learning discussion.

---

## 17. LLM-Based Relation Extraction

### Definition

An LLM can be instructed through a prompt to identify entities and relations.

Example:

```text
Extract all relations.

Return:
Entity 1 | Relation | Entity 2
```

Possible result:

```text
Steve Jobs | FOUNDED | Apple
```

### Advantages

- Flexible
- Broad pretrained knowledge
- Few-shot/zero-shot capability
- Can handle varying relation types

### Production concerns

- Cost
- Latency
- Reproducibility
- Domain suitability

### Additional context

Hallucination is an important general LLM concern, but it should not be presented as though it were a detailed lecture point unless explicitly stated in the transcript.

### Timestamp

**~01:30:00–01:40:00**, with implementation discussion around **01:55:11**.

---

## 18. Few-Shot / In-Context Learning

Conceptually:

```text
Example 1
Example 2
Example 3
      ↓
Prompt
      ↓
LLM
      ↓
New extraction
```

The model is guided by examples in the prompt rather than being fully retrained for the task.

### Timestamp

**00:25:41** and surrounding pretrained/few-shot discussion.

---

## 19. Production Trade-offs

The lecturer emphasizes that an advanced model is not automatically the best production solution.

Important considerations:

### Cost

LLM usage can be expensive.

### Latency

Production applications may require predictable low response times.

### Reproducibility

Production systems often need predictable and repeatable behavior.

### Domain suitability

Generic models may not be ideal for specialized domains.

### Deep understanding

The engineering question is:

> What is the simplest solution that satisfies the requirements?

---

## 20. Hybrid Relation Extraction

A production system can combine:

```text
Rules
+
Traditional ML
+
Deep Learning / Transformers
+
LLMs
+
Knowledge Graphs
```

Example:

```text
Known fixed pattern
       ↓
Regex/rule

Complex contextual pattern
       ↓
Transformer

Highly ambiguous/domain-specific case
       ↓
LLM

Structured validation/storage
       ↓
Knowledge Graph
```

This reflects the lecture's practical emphasis on choosing an appropriate combination of methods.

---

## 21. Domain-Specific RE

Domains such as medicine, law and finance can have specialized:

- vocabulary
- entity types
- relations
- terminology

The lecture discusses medical Relation Extraction and mentions UMLS as a medical repository.

### Exam relevance

Understand why a generic RE system may not perform equally well in every domain.

---

## 22. Multilingual RE

Two conceptual approaches:

### Translation-based

```text
Source language
 ↓
Translate to English
 ↓
English RE
```

### Direct multilingual

```text
Source language
 ↓
Multilingual RE
```

The lecture discusses Russian and also mentions work involving Indian languages.

---

## 23. Open Information Extraction

### Definition

Open IE extracts relations without requiring a fixed predefined relation schema.

Traditional schema:

```text
CURE
CAUSE
PREVENT
```

Open IE can discover relation phrases directly from text.

The lecture treats this relatively briefly.

### Exam relevance

Know the basic distinction rather than over-memorizing details.

---

## 24. Unsupervised Relation Extraction

Conceptually:

```text
Relations
 ↓
Representations
 ↓
Clustering
 ↓
Groups of similar relations
```

The lecture notes that unsupervised clustering can be noisy and is less popular than supervised and LLM-based approaches.

---

# Important Definitions

| Term | Definition |
|---|---|
| Information Extraction | Extracting structured information from unstructured text |
| Relation Extraction | Identifying semantic relationships between entities |
| Relation Classification | Determining the relation type for an entity pair |
| Relation Direction | Direction from subject/source entity to object/target entity |
| Entity Pair | Two entities considered for a possible relation |
| Rule-Based RE | RE using manually defined linguistic/pattern rules |
| Supervised RE | RE learned from labelled examples |
| Feature Engineering | Manually constructing useful model features |
| Gazetteer | Dictionary/list of known entities |
| Knowledge Graph | Structured representation of entities and relationships |
| RDF Triple | Subject–Predicate–Object representation |
| Bootstrapping | Using seed examples to generate additional examples |
| Distant Supervision | Using a knowledge graph and corpus to automatically generate training examples |
| Open IE | Relation extraction without a fixed relation schema |
| Transformer | Deep-learning architecture using attention |
| LLM | Large Language Model |
| Few-Shot Learning | Guiding a model with a small number of examples |
| In-Context Learning | Conditioning a pretrained model using information/examples in the input |
| Domain-Specific RE | RE designed for a particular domain |
| Multilingual RE | RE across multiple languages |

---

# Formulas / Rules

The lecture is mainly conceptual; there are no major mathematical derivations to memorize.

## Relation representation

```text
(Entity 1, Relation, Entity 2)
```

Example:

```text
(Albert Einstein, BORN_IN, Ulm)
```

## Classification formulation

```text
Input:
(e1, e2, context)

Output:
relation ∈ {R1, R2, ..., Rn}
```

Example:

```text
relation ∈ {CURE, PREVENT, CAUSE}
```

## Rule-based precision/recall

Conceptual rule:

```text
Rule-based RE
→ often high precision
→ potentially low recall
```

because it only captures patterns covered by the rules.

---

# Examples

## Example 1 — Basic RE

> Steve Jobs founded Apple.

```text
Steve Jobs → PERSON
Apple → ORGANIZATION

Steve Jobs --FOUNDED--> Apple
```

## Example 2 — Multiple relations

> Stanford University is located in California and was founded by Leland Stanford.

```text
Stanford University → ORGANIZATION
California → LOCATION
Leland Stanford → PERSON

Stanford University --LOCATED_IN--> California
Leland Stanford --FOUNDED--> Stanford University
```

## Example 3 — Original name

The lecture gives an IBM example involving:

> Computing, Tabulating, Recording Corporation

Conceptually:

```text
Computing, Tabulating, Recording Corporation
        |
    ORIGINAL_NAME_OF
        |
       IBM
```

## Example 4 — Medical relations

```text
Drug → TREATS → Disease

Drug → PREVENTS → Disease

Disease → CAUSED_BY → Condition
```

## Example 5 — Bootstrapping

Start:

```text
"Einstein was born in Ulm."
```

Learn:

```text
PERSON + "born in" + LOCATION
```

Use related patterns to find more examples.

## Example 6 — Distant supervision

Knowledge graph:

```text
Einstein --BORN_IN--> Ulm
```

Corpus:

> Albert Einstein was born in Ulm.

Automatically associate the sentence with `BORN_IN`.

---

# Common Misconceptions

1. **NER and RE are the same.**  
   No. NER identifies entities; RE identifies relationships.

2. **Two entities in one sentence must have a relation.**  
   No. Co-occurrence is not sufficient.

3. **RE only predicts the relation type.**  
   Not completely. Entity pair, relation and direction matter.

4. **Rule-based RE is useless.**  
   No. It can be excellent for controlled/high-precision applications.

5. **LLMs eliminate rules.**  
   No. Hybrid systems can be practical.

6. **Bootstrapping and distant supervision are identical.**  
   No. Their sources of additional training examples differ.

7. **Deep learning has no features.**  
   More accurately, it learns representations/features automatically.

8. **LLMs are always the best solution.**  
   No. Cost, latency, reproducibility and domain suitability matter.

9. **A benchmark dataset is always appropriate.**  
   No. It may not match a real application's domain.

---

# Connections to Previous Concepts

## NER → RE

```text
NER
 ↓
Entities + types
 ↓
RE
 ↓
Relations
```

## NER classification → Relation classification

Previously:

```text
Entity → PERSON / ORG / LOCATION
```

Now:

```text
Entity pair → WORKS_FOR / CAUSES / CURES / ...
```

## Feature Engineering → Deep Learning

Traditional ML relies more on manually constructed features. Deep learning learns useful representations.

## Knowledge Graph → Distant Supervision

Known KG relations can help generate training examples.

## IE → Knowledge Graph → RAG

```text
Text
 ↓
IE
 ↓
Knowledge Graph
 ↓
Graph RAG / Agentic RAG
```

## Transformers → Semantic Similarity

Contextual representations allow different surface forms with similar meanings to be related.

---

# Exam Focus

> **Assessment only:** The following priorities are my assessment based on lecture emphasis, not a claim about the actual exam paper.

## Tier 1 — Understand deeply

### 1. Relation Extraction

Be able to explain:

- What is it?
- Why is it needed?
- What is the input?
- What is the output?

### 2. NER vs RE

```text
NER:
Entity identification + entity classification

RE:
Relation existence + relation classification
```

### 3. Rule-based vs Supervised RE

| Rule-Based | Supervised |
|---|---|
| Does not require large labelled training set | Requires labelled data |
| High precision possible | Learns from examples |
| Potentially lower recall | Generalizes from training data |
| Manual rules | Learned parameters |

### 4. Bootstrapping vs Distant Supervision

Know the distinction precisely.

### 5. Knowledge Graph ↔ IE

```text
KG → helps extraction

IE → builds/populates KG
```

### 6. Feature Engineering

Know:

- POS
- dependency parsing
- words between entities
- n-grams
- head words
- entity types

### 7. Why Transformers help

Understand:

```text
Text
 ↓
Embeddings
 ↓
Contextual representation
 ↓
Attention
 ↓
Relation prediction
```

### 8. LLM production trade-offs

Remember:

```text
Cost
Latency
Reproducibility
Domain suitability
```

## Tier 2 — Important

- Domain-specific RE
- Multilingual RE
- Open IE
- Gazetteers

---

# Questions I Should Be Able to Answer

## Basic

1. What is Information Extraction?
2. What is Relation Extraction?
3. What is the difference between NER and RE?
4. Why is NER useful before RE?
5. What is an entity pair?
6. What is relation classification?
7. Why does relation direction matter?

## Rule-Based

8. What is rule-based RE?
9. What techniques can create RE rules?
10. Advantages of rule-based RE?
11. Disadvantages?
12. Why can rule-based RE have high precision?
13. Why can it have low recall?
14. When should you use rule-based RE?

## Supervised ML

15. What data is required?
16. What are relation classes?
17. What features can be used?
18. What is feature engineering?
19. How does dependency parsing help?
20. Why is labelled data expensive?
21. Why is supervised RE difficult for open-ended corpora?

## Deep Learning / Transformers

22. How is deep learning different from traditional ML?
23. What is automatic feature learning?
24. How do embeddings help?
25. Why can transformers generalize across expressions?
26. How can attention help?

## Knowledge Graphs

27. What is an RDF triple?
28. Give an RDF triple example.
29. How can IE create a knowledge graph?
30. How can a KG help IE?
31. What is the relationship between RE and Graph RAG?

## Bootstrapping

32. What is bootstrapping?
33. What are seed examples?
34. How can bootstrapping generate training data?
35. Is bootstrapping itself the final classifier?

## Distant Supervision

36. What is distant supervision?
37. How does a KG help?
38. Difference between bootstrapping and distant supervision?
39. Why is distant supervision useful?

## LLMs

40. How can an LLM perform RE?
41. What is few-shot/in-context learning?
42. Why can LLMs avoid traditional manual feature engineering?
43. Advantages of LLM-based RE?
44. Disadvantages?
45. Why might an enterprise avoid using LLMs for every RE request?

## Practical

46. When should you use regex?
47. When should you use supervised ML?
48. When should you use a transformer?
49. When should you use an LLM?
50. Why is a hybrid approach useful?
51. Why is "start simple" important?

---

# Concepts to Understand Deeply, Not Memorize

## 1. NER → RE

```text
WHO/WHAT?
    ↓
NER
    ↓
HOW CONNECTED?
    ↓
RE
```

## 2. Rule → ML → Transformer → LLM

Understand the motivation:

```text
Rules
↓
Many patterns / limited recall
↓
ML learns patterns
↓
Feature engineering becomes difficult
↓
Deep learning learns representations
↓
Transformers capture contextual relationships
↓
LLMs provide broad pretrained knowledge
```

This is a conceptual progression, not a rule that every project must follow every stage.

## 3. Bootstrapping vs Distant Supervision

```text
Bootstrapping:
small examples → more examples

Distant supervision:
knowledge graph + corpus → training examples
```

## 4. Knowledge Graph ↔ IE

```text
Text
 ↓
IE
 ↓
Knowledge Graph
 ↓
Knowledge Graph can help extraction
```

## 5. Production decision-making

The deeper lesson is:

> Understand the problem, understand the limitations of each approach, and choose the simplest approach that solves the problem.

---

# Timestamp Index

> **Important:** The transcript does not provide equally granular timestamps throughout the whole recording. Exact timestamps are retained where available. `~` indicates an approximate navigation range and is intentionally not presented as a fabricated exact timestamp.

| Timestamp | Topic |
|---|---|
| **00:02:10** | Introduction to Relation Extraction |
| **00:02:10–00:05:00** | RE vs NER vs Event Extraction |
| **~00:05:00–00:07:15** | Information Extraction tasks |
| **00:07:15** | Identifying entities and relations |
| **00:08:12–00:09:14** | Named-entity ambiguity/challenges |
| **00:09:14–00:10:00** | Complex relations and IBM example |
| **~00:10:00–00:20:00** | Relation types, entity pairs and extraction difficulties |
| **~00:20:00–00:22:00** | Rule-based RE |
| **00:22:43** | Supervised vs unsupervised discussion |
| **00:23:16** | Definition of supervised learning |
| **00:23:58–00:24:20** | Classification for RE |
| **00:24:20–00:25:00** | Relation classification |
| **00:25:02–00:25:35** | Labelled data / model learning |
| **00:25:41** | Few-shot / pretrained models |
| **00:26:38–00:27:20** | Relation tuples and RDF triples |
| **00:26:55** | Knowledge Graph ↔ IE |
| **00:27:26** | Supervised ML for RE |
| **00:28:15–00:29:09** | Rule complexity/scalability |
| **~00:30:00–00:50:00** | Supervised RE and features |
| **~00:35:00–00:45:00** | Feature engineering / dependency parsing |
| **~00:50:00–00:55:00** | Data-generation challenge |
| **~00:55:00–00:59:45** | Bootstrapping |
| **00:59:45** | Generated tuples as training data |
| **01:00:00–01:01:33** | Bootstrapping and training |
| **01:01:33–01:03:03** | Transformers / vector representations |
| **01:03:03** | Distant supervision introduction |
| **~01:03:00–01:10:00** | Distant supervision |
| **01:11:01** | Features in distant supervision |
| **01:12:16** | ML feature definition |
| **01:12:20** | Feature engineering |
| **01:13:22–01:14:27** | Relation features and classification |
| **~01:15:00–01:20:00** | Unsupervised/Open IE |
| **~01:20:00–01:25:00** | Pretrained language models |
| **~01:25:00–01:30:00** | Deep learning / transformers |
| **~01:30:00–01:35:00** | LLM-based RE |
| **~01:35:00–01:40:00** | LLM limitations/domain-specific RE |
| **~01:40:00–01:45:00** | Production approaches |
| **~01:45:00–01:50:00** | Cost, latency, reproducibility |
| **01:52:34** | Open-source implementation |
| **01:54:23** | spaCy NER implementation |
| **01:55:11** | LLM-based RE implementation |
| **01:55:11–01:56:00** | GPT/API-based extraction |
| **~01:56:00–01:58:00** | Domain-specific/multilingual RE |
| **01:58:45** | Medical RE features |
| **02:00:00** | Unsupervised RE / word embeddings |
| **02:00:46** | Unsupervised clustering discussion |

---

# Final Mental Model

```text
                     UNSTRUCTURED TEXT
                            │
                            ▼
                    Named Entity Recognition
                            │
                  ┌─────────┴─────────┐
                  │                   │
               Entity 1            Entity 2
                  │                   │
                  └─────────┬─────────┘
                            ▼
                    RELATION EXTRACTION
                            │
             ┌──────────────┴──────────────┐
             │                             │
       Is there a relation?          What type?
             │                             │
             └──────────────┬──────────────┘
                            ▼
                     RELATION TRIPLE
                            │
                  Entity ─ Relation ─ Entity
                            │
                            ▼
                     KNOWLEDGE GRAPH
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
          Graph RAG                  Agentic RAG
```

## Algorithmic evolution

```text
RULE-BASED
   │
   │ many rules / limited recall
   ▼
SUPERVISED ML
   │
   │ labelled data + feature engineering
   ▼
DEEP LEARNING
   │
   │ learned representations
   ▼
TRANSFORMERS
   │
   │ contextual representations + attention
   ▼
LLMs
   │
   │ broad pretrained knowledge / few-shot
   ▼
HYBRID SYSTEMS
   │
   ├── Rules
   ├── ML
   ├── Transformers
   ├── LLM
   └── Knowledge Graph / RAG
```

## Deepest takeaway

> **Understand the problem, understand the limitations of each approach, and choose the simplest approach that solves the problem.**
