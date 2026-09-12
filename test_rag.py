"""Tests locaux sans clé API : python -m unittest -v"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from qdrant_client import QdrantClient, models

import rag


class ModeleDeTest:
    """Vecteurs artificiels : vérifie le stockage, pas la qualité sémantique."""
    def embed(self, textes, **kwargs):
        for texte in textes:
            vecteur = np.zeros(rag.DIMENSION)
            vecteur[0] = 1
            yield vecteur


class TestRag(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            rag.COLLECTION,
            vectors_config=models.VectorParams(size=rag.DIMENSION,
                                                distance=models.Distance.COSINE))
        self.modele = ModeleDeTest()

    def tearDown(self):
        self.client.close()

    def ajouter(self, session, texte):
        return rag.indexer(self.client, self.modele, session, "test.txt", texte.encode())[0]

    def test_lecture_et_decoupage(self):
        pages = rag.lire_document("cours.md", "Bonjour été. ".encode() * 200)
        morceaux = rag.decouper(pages, "cours.md")
        self.assertGreater(len(morceaux), 1)
        self.assertTrue(all(len(m["texte"]) <= 450 for m in morceaux))
        self.assertTrue(all(m["page"] is None for m in morceaux))

    def test_fichiers_invalides(self):
        for nom, contenu in [("vide.txt", b"  "), ("image.png", b"abc"),
                              ("texte.txt", b"\xff"), ("gros.txt", b"x" * (10*1024*1024+1))]:
            with self.assertRaises(ValueError):
                rag.lire_document(nom, contenu)

    def test_isolation_meme_document(self):
        identifiant = self.ajouter("session-A", "Budget : 2400 euros")
        self.ajouter("session-B", "Budget : 2400 euros")
        self.ajouter("session-B", "Information privée de B")
        resultats = rag.rechercher(self.client, self.modele, "session-A",
                                  [identifiant], "budget")
        self.assertEqual(len(resultats), 1)
        self.assertIn("2400", resultats[0]["texte"])

    def test_reindexation_sans_doublon(self):
        self.ajouter("A", "Bonjour")
        self.ajouter("A", "Bonjour")
        self.assertEqual(self.client.count(rag.COLLECTION).count, 1)

    def test_suppression_limitee_et_documents_inactifs(self):
        identifiant = self.ajouter("A", "Document actif")
        self.ajouter("A", "Document inactif")
        self.ajouter("B", "Autre session")
        self.assertEqual(len(rag.rechercher(self.client, self.modele, "A",
                                           [identifiant], "document")), 1)
        rag.supprimer_session(self.client, "A")
        self.assertEqual(self.client.count(rag.COLLECTION).count, 1)

    def test_citations_invalides_et_absence_extraits(self):
        self.assertIn("Je ne trouve pas", rag.generer("", "", "?", [], []))
        with patch("rag.Groq") as faux_client:
            faux_client.return_value.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Budget : 2400 [9]"))])
            reponse = rag.generer("cle-test", "modele-test", "Budget ?",
                                  [{"texte": "2400", "source": "test", "page": None}], [])
            self.assertIn("n'a pas pu être validée", reponse)


if __name__ == "__main__":
    unittest.main()
