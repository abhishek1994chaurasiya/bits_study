# NLP Applications: Event Extraction - Comprehensive Study Guide
**Lecture Date:** August 8, 2026 | **Course:** NLP Applications (S2-25_AIMLCZG519)  
**Instructor:** Prof. Chetana Anoop Gavankar | **Duration:** 1h 56m 13s

---

## 1. LECTURE OVERVIEW & LEARNING OBJECTIVES

### Overview
This lecture introduces **event extraction** as a specialized form of information extraction that identifies and extracts mentions of events from unstructured text. Unlike simple entity or relationship extraction, event extraction captures *who* did *what*, *where*, and critically, *when*—making temporal information integral to the task. The lecture covers classical rule-based and ML approaches, contrasts them with modern LLM-based methods, and explores open research challenges across domains (finance, healthcare, legal, news, etc.).

### Learning Objectives
After mastering this lecture, you should be able to:
- Define event extraction and distinguish it from Named Entity Recognition (NER) and relation extraction
- Identify temporal expressions (absolute and relative) and their role in event detection
- Explain the components of an event: **trigger words**, **arguments**, and **mentions**
- Compare pipeline-based vs. end-to-end (One IE) architectures for event extraction
- Apply rule-based, ML-based, and LLM-based approaches to extract events from text
- Understand event processes, sub-events, and event relationships
- Design schema/ontology-driven extraction using generative AI methods
- Recognize cross-sentence and cross-document extraction as open research challenges

---

## 2. 5-MINUTE REVISION (QUICK REVIEW)

### Absolute Must-Know Takeaways

- **Event** = An action or occurrence at a specific time, location, or involving specific entities. Core difference from relations: events *always* have temporal information.

- **Event Components:**
  - **Trigger Word:** The verb or action-indicating term (e.g., "purchased," "acquired," "scheduled")
  - **Arguments:** Additional context (who, where, when, why) associated with the event
  - **Mention:** Complete phrase including trigger + all arguments

- **Temporal Expressions (2 types):**
  - **Absolute:** Specific dates/times (e.g., "Tuesday," "34 billion dollars," "January")
  - **Relative:** Relative to now or context (e.g., "yesterday," "last quarter," "next semester")

- **Event vs. Relation:** Event is a **specialized relation that includes time**. Example: "spouse" (relation) vs. "married on June 5th" (event).

- **Extraction Approaches:**
  - **Pipeline:** NER → Relation Extraction → Event Extraction (risk of error propagation)
  - **One IE / Parallel:** Extract NER, relations, events simultaneously in end-to-end model

- **Generative/LLM-Based:** Use **schema/ontology as input** to constrain output, preventing hallucination and ensuring domain relevance.

- **Open Challenges:** Cross-sentence/cross-document extraction, coreference resolution, event processes, multimodal data, low-resource languages.

---

## 3. DEEP-DIVE: KEY CONCEPTS

### **Concept 1: Event Extraction Fundamentals [7:07 - 9:27]**

**Formal Definition:**  
Event extraction is the process of identifying *event mentions* in text and assigning them temporal coordinates (time intervals or point-in-time), entities involved, and locations. It involves extracting *trigger words* (usually verbs) and their *arguments* to construct structured event representations.

**Intuitive Explanation:**  
> LLM Context: Imagine you're reading a news article. Instead of just extracting "Microsoft," "IBM," or "acquired" separately, you want to know: "Microsoft acquired GitHub in October 2018 for $7.5 billion." Event extraction captures this *full story* with *when* it happened baked in.

**Technical Explanation:**  
An event has three layers of structure:
1. **Trigger identification:** Finding the word that indicates an event occurred (usually a verb, but can be a noun like "acquisition").
2. **Role assignment:** Identifying which entities play which roles (acquirer, acquired item, price, date).
3. **Temporal grounding:** Linking the event to when it occurred.

The challenge is that this information may be scattered across the text, use pronouns (coreference), or be implicit.

**Formulas / Rules:**
- **Supervised classification** (for triggers): Assign each word a label (B-event, I-event, O) using features like:
  - Word identity, POS tag, lemma
  - Dependency parse relations
  - Lexical triggers (predefined lists for domain-specific events)

**Lecture Example [7:07]:**  
Prof. Gavankar stated: "We completed relation extraction last time, right?" Moving to event extraction with the example: "NLP application session is scheduled for today at 3:40 P.M." Here:
- **Trigger:** "scheduled"
- **Argument 1:** NLP application session (event being scheduled)
- **Argument 2:** today / 3:40 P.M. (temporal)

**Common Misconceptions:**
- ❌ "Events are just relations with time attached." (Partially true, but events have *structural differences* in how time is encoded and how arguments are specified.)
- ❌ "Any verb can be an event trigger." (Only certain verbs/nouns signify events; filtering by POS tags reduces false positives.)

**Memorize vs. Understand:**  
**UNDERSTAND.** You must grasp *why* temporal information matters and *how* it changes the nature of the extraction task. Rote memorization of the definition won't help; you need to recognize events in context.

---

### **Concept 2: Temporal Expressions - Absolute vs. Relative [22:35 - 25:00]**

**Formal Definition:**  
Temporal expressions are linguistic tokens that denote time. They are classified as:
- **Absolute:** Anchored to the calendar or clock (specific dates, months, years, times).
- **Relative:** Anchored to the present or to another event (e.g., "yesterday," "next semester," "two weeks later").

**Intuitive Explanation:**  
> LLM Context: Absolute time is like saying "January 1, 2026" (everyone knows what that means). Relative time is like saying "last Monday" (depends on *when* you're reading this).

**Technical Explanation:**  
The extracted temporal expression serves as a *feature* for classifying whether something is an event and for *linking* it to a timeline. This is crucial for temporal reasoning in question-answering, event ordering, and timeline construction.

**Formulas / Rules:**
Extraction methods:
1. **Rule-based/Regex:** Fixed lexicons (12 months, 7 days, common abbreviations)
2. **Feature-based ML:** Train on labeled data using lexical features, word embeddings
3. **Transformer-based:** BiLSTM or BERT fine-tuned on temporal datasets

**Lecture Examples [22:35 - 25:00]:**
- **Absolute:** "on Tuesday," "January," "34 billion dollars"
- **Relative:** "yesterday compared to today," "next semester," "last quarter," "one week workshop"
- **Trigger adjectives (temporal):** "recent," "past," "annual"
- **Time spans:** "from this date to that date"

Prof. noted: "There are only 12 months possible across the world, so you can just give those and extract them" using simple dictionaries—no need for complex NLP.

**Common Misconceptions:**
- ❌ "Temporal expressions are always dates/times." (They include duration terms like "one week," "semester," and relative descriptions.)
- ❌ "Temporal extraction requires deep learning." (Simple regex and dictionaries often suffice for months/years.)

**Memorize vs. Understand:**  
**MEMORIZE the distinction** (absolute vs. relative) because it's exam-testable. But also **UNDERSTAND** why it matters: different extraction strategies apply to each type.

---

### **Concept 3: Event vs. Relation vs. Named Entity - The Thin Line [34:45 - 39:00]**

**Formal Definition:**  
- **Named Entity:** A word or phrase referring to a discrete entity (person, organization, location, product).
- **Relation:** A semantic link between two entities (e.g., "works_for," "spouse_of," "located_in").
- **Event:** A relation that is **temporally grounded**—it specifies *when* and *where* the relation holds or occurred. Events also imply change of state or causality.

**Intuitive Explanation:**  
> LLM Context: 
> - Entity: "John Smith" 
> - Relation: "John Smith was married to Mary Johnson"  
> - Event: "John Smith married Mary Johnson on June 5, 2020"

**Technical Explanation:**  
The distinction matters for NLP systems:
- **Events are temporally specific:** "IBM purchased Microsoft" (when?) vs. "IBM was purchased by Microsoft" (completely opposite direction AND temporal specificity changes interpretation).
- **Events encode causality/state change:** Marriage is a state change; acquisition changes ownership.
- **Events have broader argument structures:** Relations typically have 2 arguments (subject, object); events may have 5+ (agent, patient, instrument, location, time, reason).

**Formulas / Rules:**
Extraction pipeline typically follows this order:
1. Identify entities (NER)
2. Identify relations among entities (Relation Extraction)
3. Identify events and their temporal anchors (Event Extraction)

**Lecture Example [34:45 - 39:00]:**
- **Relation:** "Paul Nelson killed John Smith" (who-did-what-to-whom)
- **Event:** "Paul Nelson was killed by John Smith on Tuesday, 3 PM" (adds temporal grounding and specificity)

Prof. stated: "Event is a special type of relation which involves some time associated with it."

Example: "IBM purchased Microsoft" vs. "IBM was purchased by Microsoft"—passive voice changes the agent, which is critical for event representation.

**Common Misconceptions:**
- ❌ "Relations and events are the same thing." (Events are relations + temporal + state change information.)
- ❌ "Any statement with a date is an event." (The temporal info must be *integral* to the extraction; a date mentioned in passing is not the same as an event-centric extraction.)

**Memorize vs. Understand:**  
**UNDERSTAND deeply.** The exam will likely ask you to classify statements as entities, relations, or events, or to explain why event extraction is harder than NER.

---

### **Concept 4: Event Components - Trigger, Arguments, Mention [35:00 - 41:30]**

**Formal Definition:**
An event is structured with:
- **Trigger:** The word or phrase that indicates an event occurred (typically a verb, but can be a nominalization like "meeting" or "protest").
- **Arguments:** Contextual information fulfilling semantic roles around the trigger (Agent, Patient, Instrument, Location, Time, Reason, etc.).
- **Mention:** The complete span of text including trigger and arguments that encodes a single event.

**Intuitive Explanation:**  
> LLM Context: Think of a crime report: The trigger is "robbed," and the arguments are who robbed whom, when, where, with what weapon. Together, they form the complete event mention.

**Technical Explanation:**  
For extraction, you need to:
1. Identify trigger words (by POS tag, lexicon, or learned features).
2. Extract argument spans (using sequence labeling or dependency parsing).
3. Link arguments to their roles (role classification).
4. Assemble into structured output (JSON, RDF triples, knowledge graph).

**Formulas / Rules:**
- **Trigger classification:** Softmax over event/non-event + event type
- **Argument extraction:** BiLSTM-CRF or Transformer with BIO tagging
- **Role assignment:** Classify each argument span against known role schema

**Lecture Example [35:00 - 41:30]:**
Text: "Firing of the 15 participating organizations on Tuesday."
- **Trigger:** "Firing"
- **Argument 1 (Patient):** "15 participating organizations"
- **Argument 2 (Time):** "Tuesday"
- **Argument 3 (Other):** Details about what is being fired

Prof. stated: "Trigger will be the words which are indicating that there is an event occurring. Second will be all the arguments associated with that particular event."

**Common Misconceptions:**
- ❌ "Every noun is a potential trigger." (Nouns like "table," "person," "year" are rarely triggers; events are typically indicated by verbs or specific nominalizations.)
- ❌ "Arguments must be adjacent to the trigger." (Arguments can be scattered across sentences, requiring coreference resolution.)

**Memorize vs. Understand:**  
**MEMORIZE the terminology** (trigger, argument, mention) because it appears throughout the field. **UNDERSTAND** why separating these components helps with modular error analysis and improvement.

---

### **Concept 5: Pipeline Approach & Error Propagation [48:00 - 54:30]**

**Formal Definition:**  
A **pipeline approach** to information extraction processes tasks sequentially: NER → Relation Extraction → Coreference Resolution → Event Extraction. Errors from each step are passed downstream, potentially compounding.

**Intuitive Explanation:**  
> LLM Context: Imagine an assembly line. If the first station makes a 10% error, and the second station makes a 10% error independently, the final product has ~19% error, not 10%. And this compounds with each station.

**Technical Explanation:**  
In a standard pipeline:
1. **NER module:** Achieves ~90% accuracy → outputs entities with 10% noise.
2. **Relation extraction:** Trained on gold entities but now gets 10% noisy entities → accuracy drops to ~70% on real data.
3. **Coreference resolution:** Further degrades (e.g., to ~50% effective accuracy).
4. **Event extraction:** Inherits all upstream errors.

The issue: **no feedback loop.** If NER misses an entity, Relation and Event modules have no way to recover.

**Formulas / Rules:**
Error compounding formula (approximate):
```
Final Accuracy ≈ P(NER correct) × P(Relation | NER correct) × P(Event | Relation correct)
                ≈ 0.90 × 0.70 × 0.50 ≈ 31.5% (vs. ideal ~90%)
```

**Lecture Example [48:00 - 54:30]:**
Text: "Today we have NLP application session. It is at 3:40 P.M."
- **Sentence 1 (NER):** Extracts "NLP application session" as an event.
- **Sentence 2 (Coreference):** Must realize "It" refers to the session from Sentence 1, but if NER already failed to extract "session" correctly, coreference fails.
- **Result:** Event extraction fails, and the error originated upstream.

Prof. stated: "The problem with the pipeline approach is that the error can be propagated... If there is a error in the first part, you are passing it to the next one and so on."

**Common Misconceptions:**
- ❌ "Pipelines are always worse than end-to-end systems." (Pipelines can be interpretable and modular; they're preferred when error analysis and debugging are priorities.)
- ❌ "Error compounding always makes pipelines unusable." (With very accurate individual modules, pipelines can still work well.)

**Memorize vs. Understand:**  
**UNDERSTAND the mechanism** of error propagation, as this is a classic motivation for end-to-end models in NLP. Likely exam question: "Why do researchers prefer end-to-end architectures for information extraction?"

---

### **Concept 6: One IE (End-to-End) Approach [50:30 - 58:00]**

**Formal Definition:**  
**One IE** is an end-to-end architecture that extracts Named Entities, Relations, and Events *in parallel* rather than sequentially, using a single encoder-decoder or transformer model. The key benefit: global dependencies are captured, and errors are not propagated from one module to the next.

**Intuitive Explanation:**  
> LLM Context: Instead of a relay race (one person hands to the next), it's like a team doing all tasks simultaneously and then integrating results—reducing bottlenecks and miscommunication.

**Technical Explanation:**  
Architecture:
1. **Input:** Text is converted to contextual embeddings (BERT, GPT, etc.).
2. **Encoder:** Identifies potential entities, relations, and events using learned feature weights (implicit or explicit).
3. **Parallel Classification:** 
   - Is this span a named entity? (If yes, what type?)
   - Is this relation between two entities? (If yes, what type?)
   - Is this a trigger for an event? (If yes, what type and arguments?)
4. **Decoder/Output:** Generates softmax scores and connects entities to relations to events.

**Formulas / Rules:**
Encoder-Decoder with attention:
- Input: X = [x₁, x₂, ..., xₙ]
- Encoder: H = Encoder(X) (contextual representations)
- Parallel heads:
  - P(Entity | H) = Softmax(W_ent × H)
  - P(Relation | H) = Softmax(W_rel × H)
  - P(Event | H) = Softmax(W_evt × H)

**Lecture Example [50:30 - 58:00]:**
Text: "On Tuesday, IBM acquired Red Hat for $34 billion."

- **Simultaneous extraction:**
  - **NER:** IBM (Org), Red Hat (Org), Tuesday (Date), $34B (Money)
  - **Relation:** acquired(IBM, Red Hat)
  - **Event:** ACQUISITION(acquirer=IBM, acquired=Red Hat, time=Tuesday, amount=$34B)

All in one pass, minimizing error propagation.

Prof. stated: "Here it is not capturing the dependency among the NER relation and event extraction. Here it is independently calculating the named entities relations and the event and then it is trying to connect each of them."

**Common Misconceptions:**
- ❌ "One IE requires attention mechanisms." (Attention helps but is not required; simpler architectures can also do parallel extraction.)
- ❌ "One IE always outperforms pipelines." (On gold data, pipelines can be more accurate; One IE shines on noisy real data.)

**Memorize vs. Understand:**  
**UNDERSTAND the architecture.** The exam might ask: "Compare pipeline vs. One IE for event extraction" or "Why is parallel extraction preferred?"

---

### **Concept 7: Temporal Information in Events [21:40 - 35:00]**

**Formal Definition:**  
**Temporal information** is the linguistic encoding of *when* an event occurs. It includes:
- **Point-in-time:** A specific moment (e.g., "Tuesday," "3:40 P.M.," "January 15, 2026")
- **Duration/Interval:** A span of time (e.g., "2023-2025," "5 years," "from 8 AM to 10 AM")
- **Relative temporal expression:** Relation to reference time (e.g., "yesterday," "next week")
- **Trigger adjectives:** Words like "recent," "annual," "past" that signal temporal properties

**Intuitive Explanation:**  
> LLM Context: Without temporal info, "John got married" is vague. With it: "John got married on June 5, 2020" is a concrete event. Events *demand* time grounding.

**Technical Explanation:**  
Temporal expressions are extracted using:
1. **Lexicon matching:** Dictionary of months, days, numbers.
2. **Regex patterns:** e.g., `\d{4}-\d{2}-\d{2}` for ISO dates.
3. **NER models:** BiLSTM-CRF trained on temporal entities.
4. **Temporal normalization:** "Tuesday" → need to compute absolute date based on context.

**Formulas / Rules:**
Temporal expressions can be represented in ISO 8601 format:
- Point: `2026-08-08`
- Duration: `P5Y` (5 years)
- Recurring: `XXXX-08-08` (every August 8)

**Lecture Example [21:40 - 35:00]:**
Prof. gave: "from this date to this date," "semester goes long from this date to this date," "one week conference," "one week workshop"

And: "because there are only 12 months possible across the world, so you can just give those and extract them" using a simple dictionary.

**Common Misconceptions:**
- ❌ "All temporal expressions are dates." (Durations, relative times, and recurring events are also temporal.)
- ❌ "Temporal extraction is domain-agnostic." (Different domains have different temporal conventions; medical timelines differ from financial ones.)

**Memorize vs. Understand:**  
**MEMORIZE the types** (absolute, relative, duration). **UNDERSTAND** why temporal grounding is central to event extraction.

---

### **Concept 8: Event Processes & Sub-Events [58:30 - 1:13:00]**

**Formal Definition:**  
An **event process** is a partially ordered set of sub-events that together constitute a larger, overarching event. Sub-events are:
- **Temporally ordered:** One occurs before, during, or after another.
- **Causally linked:** One sub-event may enable or cause the next.
- **Goal-oriented:** They collectively work toward achieving an end goal.

**Intuitive Explanation:**  
> LLM Context: "Getting a PhD" is not a single event; it's a 5-6 year process with sub-events: apply, interview, accept, pass qualifying exams, publish papers, write thesis, defend. Each is an event; together they form an event process.

**Technical Explanation:**  
Event processes are important for:
1. **Temporal reasoning:** Understanding *when* each sub-event occurs relative to others.
2. **Agentic AI:** An agent must plan a sequence of actions; predicting the next event in a process is crucial.
3. **Causality:** Some event sequences are causal; understanding them enables better reasoning.

**Formulas / Rules:**
Representation: Directed acyclic graph (DAG) or linear chain
```
Event₁ --[temporal/causal]--> Event₂ --[temporal/causal]--> Event₃
```

**Lecture Example [58:30 - 1:13:00]:**
**PhD Process:** Prof. detailed:
1. Check opportunities (industrial connect, full-time, part-time PhD)
2. Apply by deadline
3. Interview (based on experience/grades)
4. Selected for PhD position
5. Pass qualifying exams
6. Fulfill course requirements
7. Publish papers in A* conferences
8. Write PhD thesis
9. Defend thesis
10. Receive degree

Each is a distinct event; they have temporal constraints (order) and may have causal links.

Another example: **Buying a car:**
- Analyze existing cars
- Determine budget
- Research locations
- ... (sub-events)
- Purchase decision (main event)

**Common Misconceptions:**
- ❌ "Event processes are the same as sequences of events." (They can be, but event processes imply *structured* dependencies, not just temporal order.)
- ❌ "Only complex events have processes." (Even "buying a car" has multiple sub-events.)

**Memorize vs. Understand:**  
**UNDERSTAND deeply.** This is a burgeoning research area. Expect questions like: "What is an event process? Why is it important for agentic AI?"

---

### **Concept 9: Cross-Sentence & Cross-Document Event Extraction [1:02:30 - 1:10:00]**

**Formal Definition:**  
**Cross-sentence extraction** occurs when an event's trigger and arguments are distributed across multiple sentences. **Cross-document extraction** is when information about a single event is spread across different documents or sources.

**Intuitive Explanation:**  
> LLM Context: 
> Sentence 1: "NLP Application session is scheduled."
> Sentence 2: "It is at 3:40 P.M."
> You must realize "It" refers to the session and link the time to it.

**Technical Explanation:**  
This requires:
1. **Coreference resolution:** Identifying that pronouns/definite descriptions refer to earlier entities or events.
2. **Long-range dependencies:** Capturing relationships that span many sentences.
3. **Semantic inference:** Understanding implicit connections.

Challenges:
- **Ambiguity:** "It" could refer to multiple entities.
- **Ellipsis:** Information may be omitted (e.g., repeated roles in coordinate events).
- **Implicit arguments:** An argument may be understood from context but not explicitly stated.

**Formulas / Rules:**
Cross-sentence coreference typically uses:
- Antecedent scoring: P(Antecedent_i | Mention_j)
- Span representation: embedding of mention boundaries
- Distance features: how far apart mentions are

**Lecture Example [1:02:30 - 1:10:00]:**
Text:
```
"Today, we have NLP application session. It is at 3:40 P.M."
```
- Trigger: "have" (first sentence, indicates an event)
- Temporal argument: "3:40 P.M." (second sentence)
- **Coreference:** "It" (in sentence 2) refers to "NLP application session" (in sentence 1)
- **Linking task:** Bind the temporal expression to the event

Prof. stated: "If you have it across two sentences, okay, so event is mentioned in first sentence and suppose the time, date, temporal information is mentioned in the second sentence. So now you need to do a coreference resolution that this particular temporal information is associated..."

**Harder case (cross-paragraph):**
```
[Paragraph 1] "Today, we have NLP application session."
[Paragraph 5] "The session covered event extraction."
```
Now linking "session covered event extraction" to the NLP application session from 5 paragraphs earlier is much harder.

**Common Misconceptions:**
- ❌ "Cross-sentence extraction is trivial—just use a long context window." (Long contexts help but don't solve coreference or ambiguity.)
- ❌ "Cross-document extraction is impossible." (It's challenging but feasible with knowledge graphs and embedding-based linking.)

**Memorize vs. Understand:**  
**UNDERSTAND the challenges.** This is explicitly flagged by Prof as an open research area. Expect exam questions on why it's hard and what techniques could help.

⚠️ **FLAG:** Prof. noted this as "still an open area" in research, suggesting the field doesn't have fully solved solutions yet.

---

### **Concept 10: LLM-Based & Generative Event Extraction [1:13:00 - 1:28:00]**

**Formal Definition:**  
**Generative event extraction** uses Large Language Models or Small Language Models to extract events by:
1. Providing the text as input (prompt).
2. Providing an **event schema/ontology** that constrains the output.
3. Requesting structured output (JSON, CSV, etc.).

The key advantage: **no training required**; the LLM is guided by the schema to extract within domain expectations.

**Intuitive Explanation:**  
> LLM Context: Instead of training a model on labeled data, you tell ChatGPT: "Here's a schema for events in our domain (acquirer, acquired, price, time). Now extract events from this text in JSON format." The LLM does it, constrained by the schema.

**Technical Explanation:**  
Approach:
1. **Input:** Text + event schema/ontology + prompt
2. **LLM processing:** The model uses its learned knowledge to map text to schema
3. **Constrained decoding:** Output is restricted to valid schema elements
4. **Output:** Structured data (JSON)

**Schemas can include:**
- Entity types and relationships
- Valid argument roles for each event type
- Constraints (e.g., "price must be a currency expression")

**Formulas / Rules:**
Prompt structure:
```
[TEXT]

Event Schema:
{
  "ACQUISITION": {
    "acquirer": "Organization",
    "acquired": "Organization or Asset",
    "price": "Currency Amount",
    "time": "Date"
  }
}

Extract events in JSON format adhering to the schema above.
```

**Lecture Example [1:13:00 - 1:28:00]:**
Text: "On Tuesday, IBM acquired Red Hat for $34 billion. The company said in a statement."

Schema:
```json
{
  "ACQUISITION": {
    "acquirer": "...",
    "acquired": "...",
    "price": "...",
    "time": "..."
  }
}
```

Expected output:
```json
{
  "event_type": "ACQUISITION",
  "acquirer": "IBM",
  "acquired": "Red Hat",
  "price": "$34 billion",
  "time": "Tuesday"
}
```

Prof. noted: "You are going to give the from plus the schema as a input to your generative models... and the models, language models will extract the schema, extract the event, sorry, which is from this input text. Again, when it is extracting events, it is going to ensure that it is sticking to your event schema that you have given so that you do not get a hallucination."

**Advantages over supervised learning:**
- No labeled training data needed
- Generalizes to new schemas at runtime
- Human-interpretable prompts

**Common Misconceptions:**
- ❌ "LLM-based extraction is just asking ChatGPT." (Requires careful schema design, RAG, and constrained decoding to avoid hallucination.)
- ❌ "Generative models are always better than fine-tuned small models." (Cost, latency, and accuracy trade-offs vary by application.)

**Memorize vs. Understand:**  
**UNDERSTAND the schema-driven approach.** This is the practical method in industry today. Exam question likely: "How does providing a schema improve LLM-based event extraction?"

---

### **Concept 11: Event Extraction Challenges & Open Research [1:18:00 - 1:28:30]**

**Formal Definition:**  
**Open challenges** are unresolved problems in event extraction research. They include:
1. **Ambiguous event triggers:** Words that can be triggers in some contexts but not others.
2. **Implicit events:** Events implied but not explicitly stated.
3. **Error propagation:** Errors from upstream NLP tasks (NER, parsing) degrade event extraction.
4. **Coreference & linking:** Connecting scattered arguments to events.
5. **Domain adaptation:** Generalizing across domains without retraining.
6. **Cross-lingual extraction:** Event extraction for non-English languages, especially low-resource ones.
7. **Multimodal extraction:** Extracting events from text, images, video simultaneously.
8. **Real-time efficiency:** Extracting events with low latency.
9. **Faithfulness/grounding:** Ensuring extracted events are grounded in the source, not hallucinated.

**Intuitive Explanation:**  
> LLM Context: Event extraction is "solved" for academic benchmarks but "unsolved" for production systems that must handle diverse domains, languages, and data types.

**Technical Explanation:**  
Each challenge has implications:
- **Ambiguity:** Requires context to disambiguate (e.g., "bank" can be a location or an institution).
- **Implicit events:** Requires commonsense reasoning or domain knowledge.
- **Cross-lingual:** Needs transfer learning, multilingual embeddings, or zero-shot methods.
- **Multimodal:** Requires fusion of text, image, and video modalities.

**Formulas / Rules:**
Research direction examples:
- **Few-shot learning:** Train on 1-5 examples instead of thousands.
- **Zero-shot transfer:** Use knowledge from one domain/language for another.
- **Multimodal fusion:** Combine embeddings from different modalities.

**Lecture Examples [1:18:00 - 1:28:30]:**
Prof. listed:
- "Ambiguous event triggers"
- "Detecting implicit events"
- "Cross sentence, which we talked about"
- "Event core reference resolution"
- "Few shot, zero shot event extraction"
- "Domain adaptation for the different domains"
- "Multilingual"
- "Real time event extraction"
- "Evaluation, benchmarking"

Prof. also mentioned: "Cross-lingual event extraction... for Indian languages, it's all the more difficult. So we have so many living languages and we have so much of digital news data in so many languages, but because we may not have the facility of event extraction from a non-English kind of data, it is not possible to extract this information and use it properly."

**Common Misconceptions:**
- ❌ "Event extraction is a solved problem." (It's not; production systems struggle with real-world data.)
- ❌ "All challenges are equally important." (Some are more impactful for certain applications; prioritize by use case.)

**Memorize vs. Understand:**  
**UNDERSTAND each challenge and its implications.** The exam may ask: "What is the main open challenge in cross-document event extraction?" or "Why is cross-lingual event extraction difficult?"

---

### **Concept 12: Applications of Event Extraction [20:10 - 28:40]**

**Formal Definition:**  
Event extraction has practical applications across domains:
- **News & Journalism:** Detecting political protests, natural disasters, wars.
- **Finance & Market Intelligence:** Stock price prediction from news (bull/bear markets).
- **Security & Threat Detection:** Cyber attacks, security breaches, anomalies.
- **Healthcare:** Disease outbreaks, clinical trials, treatment timelines.
- **Legal:** Case filing deadlines, temporal constraints on legal actions.
- **Customer Feedback:** Return windows, complaint timelines, SLAs.
- **Calendar & Scheduling:** Meeting arrangements, timetable management, teacher substitution (Prof.'s school example).

**Intuitive Explanation:**  
> LLM Context: Any domain with *time-sensitive actions* benefits from event extraction. News affects stock prices *at specific times*. Legal cases have *filing deadlines*. Hospitals track *disease progression over time*.

**Technical Explanation:**  
Application architecture:
1. Crawl/ingest domain-specific documents.
2. Extract events with triggers, arguments, and temporal info.
3. Store in knowledge graph or database.
4. Apply downstream reasoning (prediction, recommendation, anomaly detection).

**Formulas / Rules:**
Example for finance:
```
Event: "Apple announced Q4 2025 earnings" (time: 2026-01-15)
Trigger: "announced"
Arguments: {company: Apple, event_type: earnings, time: 2026-01-15}
→ Downstream: Check historical correlation with AAPL stock price
```

**Lecture Examples [20:10 - 28:40]:**

1. **News/Journalism:** "Political protests, natural disasters... There are so many things happening in this area, in the news, in sports news, in medical news."

2. **Finance:** "The news will affect the stock prices, whether the shares go up and down, the bull... bear market, bull market, all of these things happen based on the events that are happening across the world."

3. **Security:** "Threat detection, cyber attacks, security breaches, online."

4. **Healthcare:** "COVID time, there was a disease outbreak... clinical trials, efficacies... cancer detection at first stage."

5. **Legal:** "You need to file the case... it has to be done in a particular time period."

6. **Customer Feedback:** "When I buy a particular product, there is a window, seven days, 14 days in which I can return a product."

7. **Scheduling (Prof.'s example [17:17]):** "In my school we use a software called Fit... it will prepare all the timetables for all the teachers and it will also enable us to find the teacher for the substitutions... if there is any teacher is on leave, that software will generate a timetable."

**Common Misconceptions:**
- ❌ "Event extraction is only for NLP research." (It's deeply practical and deployed in industry.)
- ❌ "All applications require the same extraction methods." (Domain-specific schemas and triggers are necessary.)

**Memorize vs. Understand:**  
**UNDERSTAND the breadth of applications.** The exam may ask: "Provide a domain-specific example of event extraction" or "Why is event extraction useful in the healthcare domain?"

---

## 4. EXAM FOCUS & PRACTICE

### Highly Testable Material

> **LLM Context:** Based on Prof. Gavankar's emphasis, lecture structure, and her statement that "this is application scores, right? So the name itself says NLP Applications. So it is going to be all application oriented. No direct theory questions will be asked," the exam will focus on:

1. **Practical definitions & distinction-making:** Event vs. Relation vs. NER; Absolute vs. Relative temporal expressions.
2. **Architecture choices:** Pipeline vs. One IE—why and when to use each.
3. **LLM-based schema-driven extraction:** How to use generative models for event extraction.
4. **Real-world applications & challenges:** Examples from different domains; open problems.
5. **Case studies & problem-solving:** Given a text sample, extract events; design a system for a specific domain.

> **Unlikely (based on lecture style):** Pure theory (e.g., "Prove that pipeline error is exactly α×β"), implementation details of specific neural architectures, historical research deep-dives.

### Self-Assessment Questions

**Q1: Distinction & Definition**  
*Given the text: "Microsoft acquired GitHub on June 4, 2018 for $7.5 billion. GitHub is a platform for version control."*  
(a) Identify all named entities, relations, and events.  
(b) Explain why "GitHub is a platform for version control" is *not* an event, even though it contains temporal context.  
(c) What temporal expressions are present? Classify as absolute or relative.

**Expected Answer:**
- (a) **Entities:** Microsoft, GitHub, June 4 2018, $7.5 billion. **Relation:** "is a platform for" (GitHub-type relation). **Event:** ACQUISITION(acquirer=Microsoft, acquired=GitHub, time=June 4 2018, price=$7.5B).
- (b) The statement describes a property/relation, not an event. Events imply an action or change of state occurring at a specific time. "is a platform" is a state, not a change.
- (c) **Absolute:** "June 4, 2018," "$7.5 billion" (specific values). **Relative:** None in this text.

---

**Q2: Architecture & Error Propagation**  
*A company is building an event extraction system and can choose:  
(A) Pipeline: NER (95% acc.) → Relation (90% acc.) → Event (85% acc.)  
(B) One IE: Parallel extraction with BERT fine-tuned on event data (80% end-to-end acc.)*

(a) Calculate the approximate accuracy of the pipeline approach.  
(b) Which approach would you recommend and why?  
(c) What are trade-offs between the two?

**Expected Answer:**
- (a) ~95% × 90% × 85% ≈ 72.7% end-to-end (if errors are independent, which is an approximation).
- (b) Depends on domain & requirements. If interpretability & modularity matter → Pipeline. If accuracy on noisy real data matters → One IE.
- (c) **Pipeline:** Interpretable, modular, lower data requirements for each step; but error propagation. **One IE:** Better end-to-end accuracy on real data; less interpretable; requires more labeled data.

---

**Q3: Generative Event Extraction & Schema Design**  
*You're building an event extraction system for a hospital to extract disease outbreaks. Design:  
(a) An event schema (ontology) with at least 4 argument roles.  
(b) A prompt template for a generative model (LLM).  
(c) How would you ensure the model doesn't hallucinate false events?*

**Expected Answer:**
- (a) 
```json
{
  "DISEASE_OUTBREAK": {
    "disease": "string (disease name)",
    "location": "string (geographic region)",
    "confirmed_cases": "integer",
    "time_period": "date or date range",
    "severity": "enum (low, medium, high)",
    "source_data": "string (reference)"
  }
}
```
- (b)
```
You are a medical information extraction assistant. Extract disease outbreaks from the provided text. 
Adhere strictly to this schema: {SCHEMA}
Output only JSON. Do not generate events not explicitly mentioned in the text.

Text: {TEXT}
```
- (c) 
  - Use **constraint decoding:** Only output valid schema keys.
  - Require **evidence citations:** Each argument must map to text spans.
  - Use **RAG (Retrieval-Augmented Generation):** Ground outputs in authoritative medical databases.
  - Add **confidence scores:** Flag low-confidence extractions for human review.

---

**Q4: Cross-Sentence Extraction & Coreference**  
*Given:*  
```
"The AI conference was held in Mumbai. It lasted three days. Researchers from 20 countries attended it."
```

(a) Extract the event with all arguments.  
(b) Identify coreference chains.  
(c) What challenges arise if the temporal info ("three days") were in a separate paragraph?

**Expected Answer:**
- (a) **Event:** CONFERENCE(name="AI conference", location="Mumbai", duration="three days", participants="researchers from 20 countries")
  - **Trigger:** "was held"
  - **Arguments:** location, duration, participants
- (b) **Coreference:** "It" (sentence 2) → "conference" (sentence 1); "it" (sentence 3) → "conference" (sentence 1)
- (c) **Challenge:** Longer-range coreference is harder. Models must track entity references across paragraph boundaries. Risk of associating wrong temporal info to event. Requires strong context encoding (e.g., long-context transformers).

---

**Q5: Domain Adaptation & Application Design**  
*You work at a legal firm. Your goal is to extract key events from contract documents (e.g., "Payment due by June 30, 2026," "Termination clause: 30-day notice required"). Design an event extraction system.*

(a) What are the key event types and arguments for legal contracts?  
(b) Propose an extraction approach (rule-based, ML, or LLM-based). Justify.  
(c) What domain-specific challenges might you encounter?

**Expected Answer:**
- (a) **Event types:** Payment, Termination, Renewal, Penalty, Amendment, etc.
  - **Arguments:** event_type, due_date, amount, party_responsible, penalty, etc.
- (b) **Recommended:** **Hybrid (LLM + rule-based)**
  - Use **LLM with schema** for flexible event detection (handles new event types).
  - Use **rules/regex** for dates and amounts (high precision, no hallucination).
  - Combination ensures accuracy and adaptability.
- (c) **Challenges:**
  - **Legal jargon:** "Party of the first part" = one party; requires domain knowledge.
  - **Implicit events:** "The contract shall be renewed" implies future renewal event.
  - **Temporal ambiguity:** "May 2026" could mean "in May 2026" or "is allowed to 2026."
  - **Scope:** Multiple events in one clause; parsing scope correctly is hard.

---

## 5. TIMESTAMP INDEX

A chronological, skimmable reference for quick video navigation:

| Timestamp | Topic |
|-----------|-------|
| **0:00–2:09** | Class opening, poll for slides, tea break announcement |
| **2:09–7:07** | Administrative announcements: Extra session on Aug 16, Quiz 3 deferral, Situated learning reminder, Assignment deadline |
| **7:07–7:43** | Relation extraction recap; transition to event extraction |
| **8:04–11:00** | Real-world application discussion: NER/extraction in student projects (telecom, banking, software testing) |
| **13:18–18:30** | Application relevance discussion continues; faculty substitution scheduling as use case for event extraction |
| **18:30–19:16** | Q&A on event extraction intro; muting participants |
| **19:17–20:10** | Event extraction overview begins |
| **20:10–28:40** | **[KEY]** Event extraction applications across domains: News, Finance, Security, Healthcare, Legal, Customer Feedback, Scheduling |
| **28:40–35:00** | **[KEY]** Temporal expressions (absolute vs. relative) and their role in events; temporal triggers and adjectives |
| **35:00–41:30** | **[KEY]** Event vs. Relation vs. Named Entity distinction; event components (trigger, arguments, mention) |
| **41:30–46:00** | Event extraction definition; mentioning named entities as prerequisites; subtasks of information extraction |
| **46:00–48:00** | Applications of events in prediction and comprehension; common challenges in info extraction |
| **48:00–54:30** | **[KEY]** Pipeline approach and error propagation problem |
| **54:30–58:00** | **[KEY]** One IE / End-to-End parallel approach; architecture overview |
| **58:30–1:02:30** | Event processes and PhD example as case study; sub-events and temporal constraints |
| **1:02:30–1:10:00** | **[KEY]** Cross-sentence and cross-document extraction challenges; coreference resolution |
| **1:10:00–1:13:00** | Supervised and unsupervised methods; classification algorithms for event extraction |
| **1:13:00–1:28:00** | **[KEY]** LLM-based and generative event extraction; schema/ontology-driven approach; constrained decoding |
| **1:18:00–1:28:30** | **[KEY]** Open challenges in event extraction (ambiguity, implicit events, domain adaptation, cross-lingual, multimodal) |
| **1:28:30–1:35:10** | Tools, benchmarks, and libraries (spaCy, Hugging Face, GLiNER, cloud APIs); evaluation datasets |
| **1:35:10–1:40:00** | Q&A on generative event extraction; explanation of schema and constrained decoding |
| **1:40:00–1:43:00** | Dissertation orientation and faculty support discussion; upcoming session on Aug 16 |
| **1:43:00–1:53:07** | Remaining challenges, tool comparison table; recommended resources for further learning |
| **1:53:07–1:56:07** | Final Q&A, scheduling confirmation, slide upload timeline, class closure |

---

## ADDITIONAL NOTES FOR EXAM PREPARATION

### Key Concepts to Memorize (By Frequency of Mention)
1. **Trigger**: Appears 50+ times—central concept. Memorize: "Word indicating an event occurred."
2. **Event vs. Relation**: Discussed extensively. Memorize: "Event = Relation + Temporal grounding + State change."
3. **Pipeline vs. One IE**: Compare multiple times. Know both architectures and trade-offs.
4. **Temporal expressions**: Absolute vs. relative. Know examples of each.
5. **Error propagation**: Why pipeline fails. Formula: P(final) = P1 × P2 × P3.
6. **Schema/Ontology in LLMs**: Modern approach. Know why it prevents hallucination.
7. **Cross-sentence extraction**: Open problem. Understand why coreference is hard.

### Exam-Style Questions Likely to Appear
1. **Conceptual:** "Define event extraction. How does it differ from NER and relation extraction?"
2. **Application:** "You have a document corpus from a financial news site. Design an event extraction system for market prediction."
3. **Architecture:** "Compare pipeline and One IE approaches. When would you choose each?"
4. **Problem-solving:** "Given a text, extract events with triggers, arguments, and temporal info. Identify challenges."
5. **Technology:** "Explain how a generative model with schema-guided extraction prevents hallucination."
6. **Research awareness:** "What is the main open challenge in cross-document event extraction?"

### Resources Mentioned by Prof. (For Reference, Not Required for Exam)
- **Textbook:** Juravsky & Martin (Speech and Language Processing) – Chapter on information extraction
- **Code:** Harry Potter event extraction example; various Python notebooks
- **Tools:** spaCy, Hugging Face Transformers, GLiNER, LangChain, Lang Extract (Gemini)
- **Datasets:** ACE, MAVEN, ECB+, Wiki Events
- **Articles:** LLM-based event extraction; cross-lingual methods; benchmarking

> **Prof.'s reminder:** "These are for general learning. For exam, the slides as well as the textbook are sufficient."

---

## FLAG SUMMARY

⚠️ **FLAG 1 [1:17:17]:** Cross-sentence and cross-document extraction is **explicitly flagged as an "open research area" with "no fully solved solutions."** This suggests the field is still actively researching this problem. The exam may ask about this to test awareness of state-of-the-art limitations.

⚠️ **FLAG 2 [19:37]:** Regarding proprietary vs. open-source tools: "When you are doing some proprietary data and information, you cannot use open source tools." The prof acknowledges domain/privacy constraints. In your application design, consider whether data privacy requires closed-source or federated solutions.

⚠️ **FLAG 3 [1:24:02]:** Prof. mentions that some examples were "automatically generated in my copilot" and then validated by human review. This highlights the increasing use of AI in teaching materials—take note if examples seem particularly polished or if there are subtle inconsistencies.

---

**End of Study Guide**  
*Total study time estimate: 3-4 hours (reading thoroughly)  
Recommended review cycle: Read once completely, then use Timestamp Index for deep dives on weak topics.*
