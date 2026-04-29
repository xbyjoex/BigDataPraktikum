# Big-Data Praktikum – Anomalieerkennung im 3D-Druck mit Autoencoder

## Projektbeschreibung

In der additiven Fertigung (Selective Laser Melting, SLM) werden metallische Bauteile schichtweise aus einem Pulverbett per Laser aufgeschmolzen. Qualitätsmängel lassen sich bisher erst nach vollständiger Fertigung erkennen – ein kosten- und zeitintensiver Prozess.

Ziel dieses Projekts ist es, **fehlerhafte Schichten im 3D-Druckprozess automatisch zu erkennen**, indem Bilddaten einzelner Schichten mit Machine-Learning-Methoden analysiert werden.

---

## Ziele

1. **Merkmalextraktion per Autoencoder** – Schichtbilder verschiedener Metallbauteile werden in einen niedrigdimensionalen Vektorraum (Code-Vektor) projiziert.
2. **Anomalieerkennung** – Auf Basis der Code-Vektoren werden fehlerhafte Schichten identifiziert (Pseudo Labelling).
3. **Geometrieübergreifende Erkennung** – Ein vortrainiertes ResNet-Modell soll fehlerhafte Schichten auch für unbekannte Bauteilgeometrien erkennen.
4. **Visualisierung & Evaluierung** – Korrelationen und Anomalien werden visualisiert und empirisch analysiert.

---

## Aufgaben & Meilensteine

| # | Aufgabe | Abgabe / Testat |
|---|---------|-----------------|
| 1 | **Konzeptioneller Entwurf** (ca. 4 Seiten): Architektur, Ablauf, Ziele | Testat 1 – Ende Mai |
| 2 | **Implementierung**: Autoencoder + ResNet, Vorverarbeitung der Bilddaten, Evaluierung & Visualisierung | Testat 2 – Mitte/Ende Juli |
| 3 | **Abschlusspräsentation** (10 min + 5 min Diskussion): Aufgabenstellung, Lösungsskizze, Ergebnisse | Testat 3 – Anfang August oder Ende September |

---

## Technischer Stack (geplant)

- **Sprache:** Python
- **Modelle:** Convolutional Autoencoder, vortrainiertes ResNet
- **Frameworks:** PyTorch oder TensorFlow/Keras (noch zu entscheiden)
- **Daten:** Bilddaten einzelner Druckschichten verschiedener Metallbauteile

---

## Offene Fragen (zur Klärung mit dem Dozenten)

Siehe Abschnitt [Fragenkatalog](#fragenkatalog) unten.

---

## Fragenkatalog

### Datensatz & Datenzugang
1. Wie und wo werden die Datensätze bereitgestellt (Moodle, Server, Cloud, lokaler Download)?

### Aufgabendefinition & Methodik
2. Was gilt als „Anomalie" / „fehlerhafte Schicht"? Gibt es eine formale Definition oder visuelle Beispiele?
3. Soll der Autoencoder rein unüberwacht trainiert werden (Rekonstruktionsfehler als Anomaliesignal), oder wird Pseudo-Labelling als aktiver Schritt erwartet?
4. Welches ResNet soll verwendet werden (ResNet-18/50/…)? Soll ImageNet-Pretraining genutzt werden?
5. Ist Transfer Learning (Fine-Tuning des ResNet) erwünscht, oder soll das ResNet als reiner Feature-Extractor eingesetzt werden?
6. Welche Metrik soll zur Evaluierung genutzt werden (z.B. Precision/Recall, AUC-ROC, F1)?

### Vorverarbeitung
7. Wie sollen die relevanten Bauteilbereiche in den Bildern spezifiziert werden (manuelles Cropping, Masken, automatische Segmentierung)?
8. Gibt es Referenzbilder oder Ground-Truth-Masken für die Bauteilbereiche?

### Infrastruktur & Ressourcen
9. Steht GPU-Rechenkapazität bereit (Uni-Cluster, Cloud, lokal)?
10. Gibt es Vorgaben zu verwendeten Frameworks (PyTorch vs. TensorFlow) oder Bibliotheken?
11. Soll der Code in einem bestimmten Repository (z.B. GitLab der Uni) abgelegt werden?

### Entwurfsdokument
12. Welche genauen Inhalte werden im Entwurfsdokument (Testat 1) erwartet (z.B. Architekturdiagramm, Zeitplan, Literatur)?
13. Gibt es eine Vorlage oder Formatvorgabe für das Entwurfsdokument?

### Präsentation
14. Soll die Präsentation online oder in Präsenz stattfinden?
15. Welche Ergebnistiefe wird für die Präsentation erwartet (Proof-of-Concept genug, oder müssen quantitative Benchmarks vorliegen)?
