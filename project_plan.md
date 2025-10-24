**Rolle:** Du bist ein erfahrener Post-Doc im Bereich Deep Learning, spezialisiert auf Computer Vision, Meta-Learning und ressourceneffizientes Transfer Learning (PEFT). Du bist ein Experte für die Konzeption und Durchführung rigoroser, wissenschaftlicher Experimente, insbesondere unter den Beschränkungen von Plattformen wie Google Colab Pro.

**Kontext:**
Ich plane mein MCS Deep Learning Projekt. Mein Ziel ist eine empirische Studie zur Effizienz von Meta-Learning im Vergleich zu Fine-Tuning für die Klassifizierung seltener Bilder.

**Bitte beachte:** Ich stelle die Kurs- und Lab-Unterlagen (PDFs) **NICHT** zur Verfügung. Du musst dich ausschliesslich auf meinen angehängten Projektantrag und dein allgemeines Expertenwissen über die genannten Methoden stützen.

**Kern-Constraints aus dem Projektantrag:**
1.  **Methoden-Vergleich:** Meta-Learning (spezifisch **MAML**) vs. Standard Fine-Tuning (Full FT) vs. Parameter-Efficient Fine-Tuning (spezifisch **LoRA**).
2.  **Faire Vergleichsbasis:** Alle Pipelines (MAML, FT, LoRA) starten vom **exakt selben Basis-Modell** (`M_base`). Dieses `M_base`-Modell wird *zuerst* auf einem grossen Satz allgemeiner Klassen (`C_base`) vortrainiert (Strategie 3b).
3.  **Datensatz:** iNaturalist.
4.  **Ressourcen:** Alle Experimente müssen auf **Google Colab Pro** durchführbar sein.

**Deine Aufgabe:**
Basierend auf dem unten angefügten Projektantrag und deinem Expertenwissen, erstelle einen extrem detaillierten, wissenschaftlich fundierten und praktischen **Step-by-Step-Projektplan** für die Durchführung dieser Studie. Der Plan muss als praktische Anleitung dienen und die Colab Pro-Beschränkungen rigoros berücksichtigen.

Der Plan muss die folgenden Phasen detailliert ausarbeiten:

---

### Phase 1: Setup & Fundament

1.  **Umgebung:** Empfohlene Python-Bibliotheken (PyTorch, `torchmeta` oder `learn2learn` für MAML, Hugging Face `peft` für LoRA, `tqdm`, etc.) für eine Google Colab Pro Umgebung.
2.  **Modell-Auswahl:** Schlage ein Basis-Modell vor (z.B. **ResNet-34** oder ein kleines Vision Transformer (ViT)-Modell). Begründe die Wahl präzise im Hinblick auf den Trade-off zwischen Repräsentationsfähigkeit und VRAM-Bedarf auf Colab Pro.

---

### Phase 2: Daten-Pipeline (iNaturalist)

1.  **Datensatz-Sourcing:** Schlage eine Strategie für den Umgang mit iNaturalist auf Colab Pro vor (z.B. Nutzung eines etablierten Subsets wie `iNat-Mini` oder eine Strategie zum schrittweisen Download).
2.  **Wissenschaftliche Datenpartitionierung (Kritisch!):**
    * Basierend auf Best Practices (z.B. "A Closer Look at Few-shot Classification"), schlage konkrete Anzahlen von Klassen für die folgenden drei Sets vor, die für Colab Pro machbar sind:
        * **`C_base` (Basis-Set):** Klassen für das Vortraining von `M_base` UND das Meta-Training von `M_maml`.
        * **`C_val` (Validierungs-Set):** Ein Set *neuer* Klassen (weder in `C_base` noch `C_novel`) für das Hyperparameter-Tuning *aller* Adaptionsmethoden.
        * **`C_novel` (Test-Set / "Rare Classes"):** Die finalen, "seltenen" Klassen *nur* für die Endevaluation.
3.  **Data-Loader-Implementierung:**
    * **Standard-Loader:** Für das Training auf `C_base`.
    * **Episodic Task-Loader (für MAML):** Erkläre (mit Pseudo-Code), wie $N$-way $K$-shot Tasks (Episoden) mit *Support-Set* und *Query-Set* generiert werden.
    * **Few-Shot-Loader (für FT/LoRA):** Erkläre, wie die $K$-shot Support-Sets aus `C_val` und `C_novel` erstellt werden.

---

### Phase 3: Experiment-Design & "Sweet Spot"-Matrix

1.  **Achse 1: Datenmenge (Pre-Training):** Definiere Subsets von `C_base` (z.B. 25%, 50%, 100% der Klassen), auf denen `M_base` trainiert wird, um den Einfluss der Basis-Datenmenge zu testen.
2.  **Achse 2: Datenmenge (Adaption):** Definiere die Anzahl der "Shots" $K$ (z.B. $K \in \{1, 5, 10, 20\}$).
3.  **Experiment-Matrix:** Erstelle eine Tabelle, die alle zu durchlaufenden Experimente auflistet: `(Adaptions-Methode, C_base_Größe, K_shots)`.

---

### Phase 4: Pipeline 0 - Gemeinsames Pre-Training (Basis-Modell)

1.  **Schritt 4.1:** Trainiere das Modell (aus Phase 1.2) auf den `C_base`-Subsets (aus Phase 3.1) mit einer Standard-Klassifikations-Loss.
2.  **Ergebnis:** Eine Sammlung von Basis-Modellen (z.B. `M_base_25%`, `M_base_100%`) als identische Startpunkte für Pipelines 1 und 2.

---

### Phase 5: Pipeline 1 - Meta-Learning (MAML)

1.  **Schritt 5.1 (Meta-Training):**
    * Nimm ein vortrainiertes `M_base`.
    * Trainiere es mit dem **MAML-Algorithmus** weiter. Beschreibe den MAML-Trainings-Loop (Outer Loop, Inner Loop) mit Pseudo-Code unter Einbeziehung der `torch.autograd.grad` Funktionalität (oder FOMAML als ressourcenschonende Alternative).
    * **Ergebnis:** Ein meta-trainiertes Modell (z.B. `M_maml_100%`).
2.  **Schritt 5.2 (Hyperparameter-Tuning):** Erkläre, wie MAML-spezifische Hyperparameter (z.B. `meta-lr`, `inner-lr`) durch Evaluation auf Tasks aus `C_val` optimiert werden.

---

### Phase 6: Pipeline 2 - Fine-Tuning (FT & LoRA)

1.  **Schritt 6.1 (Startpunkt):** Nimm dasselbe `M_base` aus Phase 4.
2.  **Schritt 6.2 (Hyperparameter-Tuning auf `C_val`):**
    * **Full FT:** Bestimme die optimalen Hyperparameter (Learning Rate, Epochen) durch Training auf $K$-shot-Tasks aus `C_val`.
    * **LoRA FT:** Injiziere LoRA-Adapter in `M_base`. Bestimme die optimalen Hyperparameter (LR, Epochen, LoRA-`r`) auf die gleiche Weise auf `C_val`.

---

### Phase 7: Finale Evaluation & "Sweet Spot"-Analyse

1.  **Schritt 7.1 (Durchführung):** Führe alle Experimente aus der Matrix (Phase 3) durch. Für jede Kombination:
    * **MAML:** Lade `M_maml_...`. Führe MAML-Adaption (innere Schleife) auf dem $K$-shot *Support-Set* von `C_novel` durch. Messe Metriken auf dem *Query-Set* von `C_novel`.
    * **Full FT:** Lade `M_base_...`. Führe *vollständiges* Fine-Tuning auf dem $K$-shot *Support-Set* von `C_novel` durch. Messe Metriken auf dem *Query-Set* von `C_novel`.
    * **LoRA:** Lade `M_base_...` (mit LoRA). Führe *LoRA* Fine-Tuning auf dem $K$-shot *Support-Set* von `C_novel` durch. Messe Metriken auf dem *Query-Set* von `C_novel`.
2.  **Metriken (Pro Experiment):**
    * **Performance:** Top-1 Classification Accuracy.
    * **Time Efficiency (Adaptation):** Wand-Uhrzeit (Sekunden) für den Anpassungsschritt.
    * **Resource Efficiency (Training):** GPU-Stunden/VRAM-Bedarf für die *vorgelagerte* Trainingsphase (d.h. Phase 5.1 für MAML; 0 für FT/LoRA).
    * **Resource Efficiency (Adaptation):** Peak VRAM-Bedarf während der Anpassung.
3.  **Analyse & Visualisierung (Der "Sweet Spot"):**
    * Schlage 3-4 konkrete Plots vor, um die Forschungsfrage zu beantworten (z.B. X-Achse = $K$-Shots, Y-Achse = Accuracy; X-Achse = Adaptionszeit, Y-Achse = Accuracy).

---
**Angehängter Projektantrag (Text):**
Name

Raphael Güntensperger & Marc von Rohr – Team "MetaMinds"

Working Title

An Empirical Study on Data and Resource Efficiency in Rare Image Classification

Basic Research Question

Under which conditions do Meta-Learning or Fine-Tuning offer a more efficient pathway for adapting a pre-trained vision model to novel, rare classes, considering the trade-offs between classification performance, training time, and data requirements?

Key paper(s)

•

MetaMed: Few-shot medical image classification using gradient-based meta-learning (https://doi.org/10.1016/j.patcog.2021.108111)

•

Is Pre-training Truly Better Than Meta-Learning? (https://doi.org/10.48550/arXiv.2306.13841)

•

Memory Efficient Meta-Learning with Large Images (https://proceedings.neurips.cc/paper/2021/hash/cc1aa436277138f61cda703991069eaf-Abstract.html)

•

A Unified Few-Shot Classification Benchmark to Compare Transfer and Meta Learning Approaches (https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/28dd2c7955ce926456240b2ff0100bde-Paper-round1.pdf)

•

A Closer Look at Few-shot Classification (https://doi.org/10.48550/arXiv.1904.04232)

Problem & objective (~15% of content)

Problem: The classification of rare images poses a fundamental challenge for traditional deep learning models, which rely on large, balanced datasets. In critical domains like medical diagnostics, industrial fault detection, or biodiversity monitoring, "rare" classes are often the most important, yet data for them is inherently scarce. This creates a significant bottleneck for the practical application of AI.

Objective: The primary objective of this project is to systematically compare two key training paradigms—Meta-Learning and Fine-Tuning—for the task of rare image classification. Meta-Learning is particularly important in this context because it is explicitly designed to "learn to learn" from very few examples, mirroring the real-world scenario of encountering a rare category. We aim to identify the "sweet spot"—the optimal balance of data, computational cost, and performance—to determine which method is superior under these conditions.

Data (~20% of content)

Data Source and Size: We will use the iNaturalist dataset. To systematically evaluate the impact of data scale, we will conduct experiments on subsets of varying sizes. We will analyze the dataset structure to ensure that the classes

Labels and Quality: The data is pre-labeled with high quality. For our experiments, we will partition the classes into distinct sets for meta-training/pre-training and for the final few-shot evaluation on the "rare" classes, ensuring no class overlap.

Approach (~30% of content)

Our methodology is centered on a direct comparison of two learning pipelines. To ensure a fair and scientifically valid comparison, both pipelines will start from the exact same pre-trained base model (e.g., a ResNet variant of ResNet-18/34/50). The purpose of this project is not to compare different model architectures, but to compare the effectiveness of the adaptation strategy (Meta-Learning vs. Fine-Tuning). Using the same starting point ensures that any observed performance differences are attributable to the method itself, not the underlying model. This base model will have been trained on a broad set of general classes but will not have seen the specific "rare" classes used in our evaluation.

Method 1 - Meta-Learning: We will adapt the pre-trained model for meta-learning. The model will be trained on a variety of few-shot tasks drawn from the general training classes, teaching it the skill of rapid adaptation.

Method 2: Fine-Tuning: The same base model will be adapted to the held out "rare" classes using a minimal number of examples. We will investigate and compare both regular fine-tuning and parameter-efficient fine-tuning (PEFT) techniques.

Experiments (~20% of content)

Setup: We will design a controlled experiment where both methods are tasked with classifying the same set of "rare" classes using the same number of examples ("shots"). The experiment will be repeated with varying dataset sizes for the initial training phase to analyze the relationship between data volume and final performance.

Baselines: The two methods—Meta-Learning and Fine-Tuning—will serve as direct baselines for each other. We exclude comparisons to large, closed-source models, as their training data is unknown, making a fair scientific comparison impossible.

Evaluation Metrics: The core of our project is the comparison based on a "sweet spot" analysis. We will measure and compare:

1.

Performance: Classification accuracy on the unseen rare classes.

2.

Time Efficiency: Time required for the adaptation phase.

3.

Data & Resource Efficiency: The required dataset size and computational power (e.g., in GPU hours) to achieve target performance levels.

What`s new? (~5% of content)

The novelty lies in the direct, empirical head-to-head comparison of these two powerful learning paradigms specifically for the task of rare image classification. Our work focuses on providing a practical guide to their relative strengths concerning data and resource efficiency, aiming to identify the optimal approach for this critical, data-scarce problem.

So what? (~5% of content)

The outcome will provide valuable insights for practitioners facing rare category problems. It will help answer the critical question: "Given a limited dataset and computational budget, should I invest in a meta-learning setup or a fine-tuning approach?" This has a real-world impact on making AI more effective for high-stakes problems with limited data.

Other Considerations (~5% of content)

Feasibility & Ambition: The project is realistic and ambitious, involving a rigorous comparison of two major deep learning techniques.

Risk Assessment: The risk is "low-to-moderate." The primary challenge is the careful and fair implementation of the experimental setup. Managing computational resources for multiple training runs is a key logistical challenge.

Challenges: Key tasks include researching best practices for the two adaptation methods and designing a clear data-partitioning pipeline.

Scope & Resources: The scope is well-defined. The project requires GPU access. We plan to use Google Colab Pro for our computational experiments.
