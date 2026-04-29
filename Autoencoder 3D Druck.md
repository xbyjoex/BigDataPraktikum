# Pseudo labelling und Vorhersage von Anomalien im 3D-Druck 

## Motivation
In der additiven Fertigung, insbesondere bei der Herstellung metallischer Bauteile kommt das 
Selective-Laser-Melting (SLM) zum Einsatz. Beim SLM wird ein Metallpulverbett mithilfe eines Lasers 
schichtweise aufgeschmolzen und eine dreidimensionale Fertigungsaufgabe auf zwei Dimensionen 
reduziert. Die Grundlage für SLM stellt ein CAD-Modell dar, welches das Bauteil im Vorfeld digital in 
einzelne Schichten aufteilt. Es besteht somit die Möglichkeit, auch individuell komplexe, geometrische 
Fertigungsgegenstände herzustellen. Zum aktuellen Zeitpunkt werden verschiedene Komponenten 
und Strukturen der Bauteile durch Trial-and-Error optimiert. Dies hat zur Folge, dass Bauteile erst 
nach dem Vollenden des gesamten Bauprozesses auf ihre Bauteilqualität untersucht werden können. 
Dazu werden die Bauteile durch verschiedene Experimente auf Zug -und Druckfestigkeit geprüft, da 
dies ausschlaggebende Faktoren für die Bauteilqualität sind. Dieser Prozess ist sehr kostenintensiv, 
zeitaufwendig und im Allgemeinen sehr ineffizient. Mithilfe von maschinellen Lernmethoden können 
Sensordaten schon während der Fertigung analysiert und ausgewertet werden. Auf diese Weise 
lassen sich Rückschlüsse, hinsichtlich relevanter Prozesseigenschaften und Produktqualitätseinflüsse 
ableiten.  

## Zielstellung  
Anhand von Bilddaten einzelner Schichten verschiedener Metallbauteile mit unterschiedlichen Geometrien sollen Merkmale 
charakterisiert und mögliche Korrelationen bzw. Anomalien im Druckprozess aufgezeigt werden. 
Dabei sollen Autoencoders zum Einsatz kommen, mit denen die Merkmale in den Bilddaten 
zunächst in einen niedrigdimensionalen Vektorraum projiziert werden. Auf Grundlage dieser 
projizierten Code Vektoren lassen sich fehlerhafte Schichten ableiten. 
Mithilfe einer vortrainierten Resnet-Architektur sollen diese fehlerhaften Schichten anschließend Geometrieübergreifend erkannt werden.  

## Aufgaben

### 1. Theoretische Grundlagen und Lösungsskizze: 

Im ersten Schritt sollen die theoretischen Grundlagen eines Autoencoders erarbeitet und verstanden 
werden. Weiterhin sollen Implementierungsmöglichkeiten mithilfe von 
vorhanden Python Bibliotheken recherchiert werden. Hinsichtlich der Bilddaten ist eine 
Vorverarbeitung notwendig, bei der die jeweiligen Bereiche der Bauteile zu spezifizieren sind. 

### 2. Implementierung: 

Auf Basis der Lösungsskizze ist ein Autoencoder und ein vortrainiertes RESNET selbstständig zu implementieren und auf die 
gegebenen Datensätze anzuwenden. 

Evaluierung und Visualisierung: 
Die implementierten Funktionen sind auf den Datensätzen zu evaluieren. Korrelationen 
und Anomalien zwischen den Bauteilen sollen visualisiert und durch empirische Analysen 
verdeutlicht werden.

### 3. Vortrag: 
Ausarbeitung einer 10-minütigen Präsentation, in welcher die Aufgabenstellung, die Lösungsskizze 
(Beschreibung der Verfahren, Anwendung, sowie Nennung der verwendeten Bibliotheken) und die 
Ergebnisse der Evaluierung dargestellt werden. 
