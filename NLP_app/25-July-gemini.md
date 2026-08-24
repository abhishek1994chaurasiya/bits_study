# Lecture Overview
**Course:** NLP Applications (S2-25_AIMLCZG519)
**Professor:** Chetana Anoop Gavankar
**Date:** July 25, 2026
**Topic:** Information Extraction (IE) and Named Entity Recognition (NER)

This lecture introduces the broad field of Information Extraction with a specific deep dive into Named Entity Recognition (NER). It covers the IE pipeline, challenges with different data formats, and the evolution of NER approaches ranging from rule-based and machine learning (MEMM, CRF) to neural networks (Bi-LSTM, Transformers) and LLM/RAG setups.

# Learning Objectives
* Understand the definition and scope of Information Extraction (IE).
* Identify the key components of the IE pipeline.
* Grasp the subtasks of Named Entity Recognition (NER): finding and classifying entities.
* Evaluate and compare different approaches to NER (Rule-based, ML-based, Deep Learning, and LLMs).
* Understand labeling strategies like BIO/IOB and BILOU encoding.
* Recognize real-world implementation challenges and the role of tools like spaCy.

# 5-Minute Revision
* **Information Extraction:** Automatically identifying structured data (entities, relations, events) from unstructured text.
* **IE Pipeline:** NER -> Relation Extraction -> Event Detection -> Template Filling. 
* **NER (Named Entity Recognition):** Extracting proper nouns or key information (Person, Org, Location, Date, Money) from text.
* **BIO Tagging:** Standard labeling format for sequence prediction (Begin, Inside, Outside).
* **Approaches:**
    * **Rule-based:** Uses RegEx/dictionaries. Zero training cost, highly explainable, but hard to scale.
    * **Machine Learning:** HMM, MEMM, CRF. Uses feature engineering (POS tags, word shape, previous words). Data-hungry.
    * **Deep Learning (Bi-LSTM-CRF):** Learns representations automatically. Bi-LSTM captures left and right context; CRF enforces transition rules between tags (e.g., an 'I-Person' must follow a 'B-Person').
    * **Transformers (BERT):** Attention mechanism captures long-range dependencies across the text, making disambiguation highly accurate.
* **Real-world Application:** Often requires a hybrid approach (Rules for simple patterns, Transformers for complex entities, Knowledge Graphs/RAG for domain-specific entity verification).

# Key Concepts

### 1. Information Extraction (IE) & IE Pipeline
*(Explicitly taught in lecture)*
*   **Definition:** Automatically identifying pieces of relevant, structured information from unstructured or semi-structured text. 
*   **Pipeline Stages:** NER (identifying entities), Relation Extraction (identifying relationships between entities), Coreference Resolution (linking pronouns to entities), Event Detection (capturing temporal data/events), and Template Filling (populating databases or knowledge graphs).

### 2. Named Entity Recognition (NER) 
*(Explicitly taught in lecture)*
*   **Definition:** Identifying and categorizing key information (entities) in text into predefined classes (Person, Location, Organization, Medical codes, etc.).
*   **Subtasks:** 1) Segmentation (finding the entity span) and 2) Classification (assigning the entity type).

### 3. NER Tagging Standards (BIO / BILOU)
*(Explicitly taught in lecture)*
*   **Definition:** Formats used to label sequential data for model training. 
*   **BIO:** 'B' marks the beginning of an entity, 'I' marks the inside/continuation, and 'O' means outside (not an entity).

### 4. Approaches to NER
*(Explicitly taught in lecture)*
*   **Rule-Based:** Uses regular expressions and gazetteers (dictionaries).
*   **Machine Learning (MEMM & CRF):** Treats NER as a sequence classification problem using engineered features.
*   **Neural Models (Bi-LSTM & Bi-LSTM-CRF):** Uses deep sequence modeling to capture bidirectional context without manual feature engineering.
*   **Transformers & LLMs:** Uses self-attention to manage long-term dependencies. Generative models (LLMs) can extract entities zero-shot but are costly.

# Detailed Explanations

### Concept: BIO and BILOU Tagging
*   **Intuitive Explanation:** If you are highlighting text, you need a way to tell the computer exactly where the highlight starts and stops, especially if the name is three words long (like "Ratan Tata Foundation").
*   **Technical Explanation:** BIO tagging solves the span boundary problem in sequence labeling. `B-ORG` denotes the first word of an organization. `I-ORG` denotes subsequent words. `O` denotes non-entities. BILOU adds `L` (Last) and `U` (Unit/Single word) to give the model even finer-grained span boundaries, avoiding the need to look ahead to determine if an entity has ended.
*   **Simple Example:** 
    *   Ratan (B-ORG)
    *   Tata (I-ORG)
    *   Foundation (I-ORG)
    *   visited (O)
    *   Mumbai (B-LOC)
*   **Exam Relevance:** High. Understanding how to manually tag a sentence using BIO format is a classic exam question. *(My Assessment)*
*   **Timestamps:** 40:00, 1:10:00

### Concept: Maximum Entropy Markov Model (MEMM) vs. Conditional Random Fields (CRF)
*   **Intuitive Explanation:** MEMM makes decisions one step at a time looking only backwards, which can lead it into a trap (label bias). CRF looks at the whole sequence globally before deciding the best path of tags.
*   **Technical Explanation:** MEMM is a discriminative model that calculates the probability of the current tag given the previous tag and the input features. However, greedily predicting step-by-step can lead to suboptimal sequences. CRFs calculate conditional probabilities globally over the entire sequence, ensuring the sequence of tags makes structural sense (e.g., preventing an `I-ORG` tag from appearing right after an `O` tag).
*   **Ambiguity Flag:** The professor briefly mentions beam search versus greedy search for MEMM (52:33 - 57:00). *[Assistant's context: Beam search explores the top 'k' most likely sequence paths rather than strictly taking the single highest probability at each step, mitigating MEMM's greedy flaws.]*
*   **Exam Relevance:** Medium. You should understand *why* CRF is preferred over standalone MEMM or HMM. *(My Assessment)*
*   **Timestamps:** 52:33, 1:02:58

### Concept: Bi-LSTM-CRF Architecture
*   **Intuitive Explanation:** Bi-LSTM reads the sentence from left-to-right and right-to-left to understand the surrounding context of a word. The CRF layer acts as a grammar checker for the output tags to make sure they follow sequence rules.
*   **Technical Explanation:** The Bi-LSTM creates hidden state representations of each token incorporating past and future context. Instead of just outputting the softmax probabilities for each token independently, the Bi-LSTM passes these representations to a CRF layer. The CRF layer calculates transition scores between tags, conditioning the final output sequence on both the input text and the legal transitions between tags.
*   **Exam Relevance:** High. This was a state-of-the-art baseline before transformers and is heavily utilized in industry. *(My Assessment)*
*   **Timestamps:** 1:16:48

### Concept: Transformer-based NER
*   **Intuitive Explanation:** Transformers look at every word in the sentence simultaneously to figure out which words are most relevant to each other, no matter how far apart they are.
*   **Technical Explanation:** Encoder-only transformers (like BERT) utilize multi-headed self-attention to capture long-range dependencies and coreferences. This allows the model to deeply understand context. For example, if a pronoun refers back to an entity much earlier in the text, the attention mechanism captures this relationship natively, drastically improving extraction accuracy over LSTMs.
*   **Exam Relevance:** High. Expect to differentiate why Transformers are better than Bi-LSTMs for NER. *(My Assessment)*
*   **Timestamps:** 1:21:00

# Important Definitions
*   **Coreference Resolution:** Identifying when different words (e.g., "Chetana", "She", "The professor") refer to the same entity in a text. *(Explicitly Taught)*
*   **Event Detection:** A specialized form of relation extraction focusing on temporal metadata (e.g., extracting the start and end dates of a conference). *(Explicitly Taught)*
*   **Gazetteer:** A predefined dictionary or list of names (e.g., all countries, known genes) used in rule-based NER systems to match entities. *(Explicitly Taught)*

# Formulas / Rules
*   **Feature Engineering Rules for ML-based NER:** Features typically include the current word, previous word, next word, POS tags, capitalization (First letter cap, all caps), word length, and internal punctuation (e.g., hyphens for medical terms). 
*   *Note: Mathematical formulas for MEMM and CRF were shown on slides but the professor explicitly stated she will not ask mathematical problems on CRF in the exam (Timestamp: 1:02:58).*

# Examples
*   **Ambiguity Example:** "Washington" (Could be a Person or a Location). "May" (Could be a Person or a Month). "Apple" (Could be a fruit or an Organization). *(From Professor)*
*   **Complex Span Example:** "Birla Institute of Technology and Science, Pilani" - It's difficult to determine where the entity boundaries start and stop without robust context features. *(From Professor)*

# Common Misconceptions
*   **Misconception:** LLMs (like GPT) and Transformers are the exact same thing. 
    *   **Correction:** Transformers are the underlying neural network *architecture* (which includes Encoders and Decoders). LLMs are specific *applications/pre-trained models* (usually Decoder-only like GPT, or Encoder-only like BERT) built using that architecture. *(Explicitly discussed at 1:23:31)*
*   **Misconception:** You should always use generative LLM Agents for Information Extraction.
    *   **Correction:** Using Agentic AI or massive LLMs for basic NER is an "overkill" (high compute cost, slow). Industry often uses smaller Encoder models (BERT), Bi-LSTM, or even Regex rules for specific highly-structured data (like invoice numbers). *(Explicitly discussed at 1:31:00)*

# Connections to Previous Concepts
*   **POS Tagging:** NER is fundamentally a sequence labeling problem, exactly like Part-of-Speech (POS) tagging covered in earlier lectures. Models like HMM and MEMM are used for both. 
*   **Language Modeling:** The professor connected the predictive nature of ML models back to n-gram statistical language modeling from previous foundations.

# Exam Focus
*[My Assessment - Not stated as fact]*
*   **Deep Understanding (Do not just memorize):** 
    *   Why CRF is appended to Bi-LSTM (understanding tag transition dependencies).
    *   The role of attention in Transformers for solving long-range dependencies in IE.
    *   The difference between segmentation and classification in NER.
*   **Memorization:** 
    *   BIO / BILOU tagging mechanics.
    *   Specific feature types used in ML-based NER (capitalization, POS, previous word).
*   **Likely Exam Topics:** Given the explicit mention at 1:02:58, you will *not* need to calculate CRF math, but you *will* likely be asked to compare the architectural benefits of Bi-LSTM vs. Transformer for NER, or to manually label a sentence using BIO tagging.

# Questions I Should Be Able to Answer
1. What is the difference between Information Extraction and standard text classification?
2. If given a sentence like "Tim Cook visited Apple Park in California", how would you tag it using BIO format?
3. Why does MEMM suffer from the label bias problem compared to CRF?
4. Explain how a Bi-LSTM-CRF model works end-to-end for extracting entities.
5. Why is an encoder-only model (like BERT) preferred over a generative model (like GPT) for standard NER tasks?
6. When would you use a Rule-based NER system over a Neural Network? 

# Timestamp Index
*   **06:47** - Definition of Information Extraction (IE)
*   **08:28** - Introduction to Named Entity Recognition (NER)
*   **14:36** - The IE Pipeline (Segmentation vs. Classification)
*   **27:48** - Ambiguity and Challenges in NER
*   **33:27** - Coreference Resolution and Event Detection
*   **35:03** - Rule-based Approaches (RegEx, Gazetteers)
*   **39:00** - Machine Learning Approaches & BIO Encoding
*   **52:33** - Maximum Entropy Markov Models (MEMM)
*   **01:02:58** - Conditional Random Fields (CRF) & Exam Hints
*   **01:12:00** - Bi-LSTM Architecture for NER
*   **01:16:48** - Bi-LSTM-CRF Architecture
*   **01:21:00** - Transformers and Attention for Long-Range Dependencies
*   **01:23:31** - Difference between LLMs and Transformers
*   **01:31:00** - RAG for NER and Domain-Specific Extraction
*   **01:37:00** - Real-world hybrid/ensemble NER systems
*   **01:40:00** - Implementation (spaCy, Beautiful Soup)
*   **01:50:46** - Fine-tuning Custom Transformers vs Pre-trained Libraries
