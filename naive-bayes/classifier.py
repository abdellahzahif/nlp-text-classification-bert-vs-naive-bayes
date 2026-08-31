#!/usr/bin/env python

import sys, os, argparse, json


"""
  "News Classifier" 
  -------------------------
  This is a small interface for document classification. 
  Implement your own Naive Bayes classifier by completing 
  the class 'NaiveBayesDocumentClassifier' below.

  To run the code, 

  1. place the files 'train.json' and 'test.json' in the current folder.

  2. train your model on 'train.json' by calling > python classifier.py --train 

  3. apply the model to 'test.json' by calling > python classifier.py --apply

"""


class NaiveBayesDocumentClassifier:

    
    #vorbereitung: Erstellt leeren Ordner self.model in dem später gelerntes wissen abgespeichert wird
    def __init__(self): 

        """ The classifier should store all its learned information
            in this 'model' object. Pick whatever form seems appropriate
            to you. Recommendation: use 'pickle' to store/load this model! """
        self.model = None

        


    #liest die alten Zeitungsartikeln durch und zählt genau nach welche Wörter wie oft in welchen Kategorien vorkommen
    #daraus werden Prozentwerte berechnet und speichert dieses wissen
    def train(self, features, labels):
        """
        trains a document classifier and stores all relevant
        information in 'self.model'.

        @type features: dict
        @param features: Each entry in 'features' represents a document
                         by its so-called bag-of-words vector. 
                         For each document in the dataset, 'features' contains 
                         all terms occurring in the document and their frequency
                         in the document:
                         {
                           'doc1.html':
                              {
                                'the' : 7,   # 'the' occurs seven times
                                'world': 3, 
                                ...
                              },
                           'doc2.html':
                              {
                                'community' : 2,
                                'college': 1, 
                                ...
                              },
                            ...
                         }
        @type labels: dict
        @param labels: 'labels' contains the class labels for all documents
                       in dictionary form:
                       {
                           'doc1.html': 'arts',       # doc1.html belongs to class 'arts'
                           'doc2.html': 'business',
                           'doc3.html': 'sports',
                           ...
                       }
        """





        #6.1: Werkzeuge importieren
        from collections import Counter, defaultdict #counter zählt dinge, defaultdict verhindert abstürze bei fehlenden Schlüsseln
        import pickle #speichert fertiges Modell als datei





        #6.2: Datenanalyse
        gesamt_docs = len(labels) #zählt wie viele dokmente es insgesamt gibt
        klassen_zaehler = Counter(labels.values()) #zählt wie viele dokumente zu welcher kategorie gehören
        
        print("\nErgebnisse Aufgabe 6.2")
        for kat, anz in sorted(klassen_zaehler.items()): #geht alle kategorien alphabetisch durch
            print(f"Kategorie: {kat} -> Dokumente: {anz}") #druckt kategorie und anzahl ihrer dokumente





        #6.3: Training
        p_y = {} #leeres dictionary für grundwahrscheinlichkeiten der klassen
        for kategorie, anzahl in klassen_zaehler.items(): #geht jede kategorie durch
            p_y[kategorie] = anzahl / gesamt_docs #berechnet anteil der kategorie an allen dokumenten

        wort_zaehler_pro_klasse = defaultdict(lambda: Counter()) #erstellt Tabelle um Wörter pro Kategorie zu zählen
        for doc_id, doc_words in features.items(): #geht jedes einzelne dokument durch
            kategorie = labels[doc_id]  #holt richtige Kategorie für das aktuelle Dokument
            for wort in doc_words.keys(): #geht jedes Wort durch das im Dokument vorkommt (Häufigkeit wird ignoriert)
                wort_zaehler_pro_klasse[kategorie][wort] += 1 #erhöht zähler für dieses Wort in dieser Kategorie um 1

        vokabular = set() #erstellt eine leere sammlung für alle einzigartigen Wörter (ohne Dopplungen)
        for doc_words in features.values(): #geht alle WörterListen der Dokumente durch
            vokabular.update(doc_words.keys()) #fügt die Wörter zum globalen Vokabular hinzu

        p_x_bedingt = defaultdict(dict) #erstellt Tabelle für bedingten Wortwahrscheinlichkeiten
        for kategorie in klassen_zaehler.keys(): #geht jede Kategorie durch
            n_y = klassen_zaehler[kategorie] #holt Gesamtzahl der Dokumente in dieser Kategorie
            for wort in vokabular: #geht jedes Wort aus dem gesamten Vokabular durch
                wort_vorkommen = wort_zaehler_pro_klasse[kategorie][wort] #schaut nach wie oft das Wort in der Kategorie vorkam
                p_x_bedingt[kategorie][wort] = wort_vorkommen / n_y #berechnet die relative Wahrscheinlichkeit für das Wort

        #speichert alle berechneten Werte zusammen in einem Objekt ab
        self.model = {
            'p_y': p_y,
            'p_x_bedingt': p_x_bedingt,
            'vokabular': vokabular
        }

        with open('model.pkl', 'wb') as f: #öffnet eine Datei namens model.pkl zum Schreiben im Binärmodus
            pickle.dump(self.model, f) #schreibt das gesamte ModellDictionary in diese Datei

        print("\nModell erfolgreich gelernt und in 'model.pkl' gespeichert.")


        #Hilfsausgabe für die Tabellen
        print("\n=== WERTE FÜR TABELLE 1: P[class] ===")
        for klasse, wahrscheinlichkeit in sorted(p_y.items()): #geht alle Kategorien sortiert durch
            print(f"Klasse: {klasse:<12} -> P[class]: {wahrscheinlichkeit:.4f}") #druckt berechneten KlassenPrior aus

        print("\n=== WERTE FÜR TABELLE 2: P[class][word] ===")
        #liste mit den genauen WortKlassenPaaren die auf dem Aufgabenblatt gesucht werden
        gesuchte_woerter = [
            ("app", "technology"), ("arriv", "travel"), ("art", "arts"),
            ("compani", "business"), ("flavor", "dining"), ("market", "business"),
            ("prevent", "health"), ("run", "sports"), ("tour", "travel"), ("world", "travel")
        ]
        for wort, klasse in gesuchte_woerter: #geht die gesuchten Paare durch
            wert = p_x_bedingt[klasse].get(wort, 0.0) #holt die Wahrscheinlichkeit ->gibt 0.0 zurück falls das Wort nie vorkam
            print(f"Wort: {wort:<8} | Klasse: {klasse:<12} -> P[class][word]: {wert:.4f}")

        # FIXME: implement training

        # FIXME: at the end of training, store self.model using pickle.








    #Anwenung: bekommt neue, unbekannte Artikel
    #schaut nach welche Wörter im text drin sind und welche fehlen
    #vergleicht das mit dem gelernten Wissen aus dem trainig und rät dann die logischste Kategorie für jeden artikel        
    def apply(self, features):
        """
        applies a classifier to a set of documents. Requires the classifier
        to be trained (i.e., you need to call train() before you can call apply()).

        @type features: dict
        @param features: see above (documentation of train())

        @rtype: dict
        @return: For each document in 'features', apply() returns the estimated class.
                 The return value is a dictionary of the form:
                 {
                   'doc1.html': 'arts',
                   'doc2.html': 'travel',
                   'doc3.html': 'sports',
                   ...
                 }
        """
        import pickle
        import math #importiert Mathe Module für die logarithmus Funktion

        epsilon = 1e-1 #hier werden die testwerte eingetragen

        #falls das Modell noch nicht im Arbeitsspeicher ist wird es aus der Datei geladen
        if self.model is None:
            with open('model.pkl', 'rb') as f: #öffnet gespeicherte Datei zum Lesen im Binärmodus
                self.model = pickle.load(f) #lädt modell wieder in den Arbeitsspeicher

        #entpackt gelernten Daten aus Modell Objekt
        p_y = self.model['p_y']
        p_x_bedingt = self.model['p_x_bedingt']
        vokabular = self.model['vokabular']

        vorhersagen = {} #erstellt leeres dictionary dür fertigen Text Vorhersagen

        #jedes Dokument im Testset klassifizieren
        for doc_id, doc_words in features.items(): #Schleife über alle dokumente im Testset
            beste_klasse = None #var für die am besten passende Kategorie
            bevorzugter_wert = -float('inf') #startwert auf minus unendlich setzen um maximum zu finden

            #Wahrscheinlichkeit für jede der 8 Klassen berechnen
            for klasse in p_y.keys(): #geht alle 8 Kategorien durch 
                #rechnen mit Logarithmen wegen numerischer Stabilität -> verhindert dass die Zahlen zu klein werden und auf 0 runden(underflow verhindern)
                log_wahrscheinlichkeit = math.log(p_y[klasse]) #startwert ist Logarithmus der Klassenwahrscheinlichkeit

                for wort in vokabular: #geht jedes gelernte Wort durch (Bernoulli-Modell prüft das gesamte Vokabular)
                    wort_vorhanden = wort in doc_words #prüft boolesch -> ist das Wort im aktuellen Dokument enthalten? True/False

                    prob = p_x_bedingt[klasse][wort] #holt gelernte Wahrscheinlichkeit für dieses Wort in dieser Klasse

                    #7.1: epsilon Glättung
                    if wort_vorhanden: #Fall A -> Wort ist im dokument enthalten
                        #falls Wahrscheinlichkeit 0.0 ist wird sie durch das kleine epsilon ersetzt
                        glättung_prob = max(prob, epsilon) 
                        log_wahrscheinlichkeit += math.log(glättung_prob)
                    else:  #Fall B -> Wort ist nicht im Dokument enthalten
                        #falls Gegenwahrscheinlichkeit 0.0 ist wird sie durch das kleine epsilon ersetzt
                        glättung_gegen_prob = max(1 - prob, epsilon) 
                        log_wahrscheinlichkeit += math.log(glättung_gegen_prob)

                #höchsten Score ermitteln
                if log_wahrscheinlichkeit > bevorzugter_wert: #wenn die aktuelle Klasse wahrscheinlicher ist als die bisher beste
                    bevorzugter_wert = log_wahrscheinlichkeit #aktualisiert den besten Wert
                    beste_klasse = klasse #merkt sich diese Klasse als neue beste Klasse

            vorhersagen[doc_id] = beste_klasse #speichert die ermittelte beste Klasse für dieses Dokument ab

        return vorhersagen, epsilon #gibt Vorhersagen und das benutzte epsilon zurück

        # FIXME: implement the model application


                
if __name__ == "__main__":

    # parse command line arguments (no need to touch)
    parser = argparse.ArgumentParser(description='A document classifier.')
    parser.add_argument('--train', help="train the classifier", action='store_true')
    parser.add_argument('--apply', help="apply the classifier (you'll need to train or load"\
                                        "a trained model first)", action='store_true')
    parser.add_argument('--inspect', help="get some info about the learned model",
                        action='store_true')

    args = parser.parse_args()

    classifier = NaiveBayesDocumentClassifier()

    def read_json(path):
        with open(path) as f:
            data = json.load(f)['docs']
            features,labels = {},{}
            for f in data:
                features[f] = data[f]['tokens']
                labels[f]   = data[f]['label']
        return features,labels
    
    if args.train:
        features,labels = read_json('train.json')
        classifier.train(features, labels)

    if args.apply:
        features,labels = read_json('test.json')
        result, aktuelles_epsilon = classifier.apply(features)



        #Fehlerrate berechnen
        falsch_klassifiziert = 0 #Zähler für die fehler
        gesamt_test_docs = len(labels) #Gesamtanzahl der Testdokumente

        for doc_id, wahre_klasse in labels.items(): #schaut sich jedes dokument und dessen echte Kategorie an
            vorhergesagte_klasse = result.get(doc_id) #holt die vorhersage ders Klassifikators
            if vorhergesagte_klasse != wahre_klasse: #wenn die vorhersage falsch war
                falsch_klassifiziert += 1 #erhöht den fehlerzähler um 1

        fehlerrate = (falsch_klassifiziert / gesamt_test_docs) * 100 #berechnet prozentualen anteil der fehler
        print(f"\nKlassifikation abgeschlossen.")
        print(f"Genutzter Epsilon-Wert: {aktuelles_epsilon} ({aktuelles_epsilon:.0e})")
        print(f"Gesamtanzahl Dokumente im Testset: {gesamt_test_docs}")
        print(f"Falsch klassifiziert: {falsch_klassifiziert}")
        print(f"Fehlerrate (Error Rate): {fehlerrate:.2f}%") #gibt fehlerrate aus



            
    
