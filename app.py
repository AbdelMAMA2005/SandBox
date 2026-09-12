"""L'interface. Lancer avec : python -m streamlit run app.py"""
import hashlib
import os
import time
from uuid import uuid4

import streamlit as st
from fastembed import TextEmbedding
from groq import AuthenticationError, RateLimitError

from rag import (EMBEDDING_MODEL, connecter_qdrant, generer, indexer,
                 rechercher, supprimer_session)

st.set_page_config(page_title="Mon chatbot documentaire", page_icon="📚")
st.title("📚 Mon chatbot documentaire")
st.write("Importe tes documents, pose une question et vérifie les sources.")


def parametre(nom, defaut=""):
    valeur = os.environ.get(nom)
    if valeur:
        return valeur
    try:
        return st.secrets.get(nom, defaut)
    except FileNotFoundError:
        return defaut


@st.cache_resource
def charger_modele():
    # Seul le modèle est partagé ; aucun document n'est mis dans ce cache.
    return TextEmbedding(model_name=EMBEDDING_MODEL, threads=2)


@st.cache_resource
def charger_base():
    return connecter_qdrant(parametre("QDRANT_URL"), parametre("QDRANT_API_KEY"))


def afficher_sources(sources):
    if sources:
        with st.expander("Voir les extraits consultés"):
            for i, source in enumerate(sources, start=1):
                page = f" — page {source['page']}" if source.get("page") else ""
                st.text(f"[{i}] {source['source']}{page}")
                st.text(source["texte"])


for nom in ("GROQ_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"):
    if not parametre(nom):
        st.info(f"Configuration manquante : {nom}. Consulte le README.")
        st.stop()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
    st.session_state.documents = {}
    st.session_state.messages = []
    st.session_state.dernier_appel = 0.0

try:
    client = charger_base()
except Exception:
    st.error("Connexion à Qdrant impossible. Vérifie l'URL, la clé et l'état du cluster.")
    st.stop()

with st.sidebar:
    st.header("Mes documents")
    st.caption("PDF avec texte, TXT ou MD en UTF-8. 5 documents par session, 10 Mo chacun.")
    st.caption("Les extraits sont stockés dans Qdrant et les passages consultés sont envoyés à Groq.")
    fichiers = st.file_uploader("Choisis tes fichiers", type=["pdf", "txt", "md"],
                                accept_multiple_files=True, key=st.session_state.session_id)
    if st.button("Analyser les documents", disabled=not fichiers):
        for fichier in fichiers:
            contenu = fichier.getvalue()
            empreinte = hashlib.sha256(contenu).hexdigest()
            if empreinte in st.session_state.documents:
                st.info(f"Déjà analysé : {fichier.name}")
                continue
            if len(st.session_state.documents) >= 5:
                st.warning("Maximum 5 documents. Supprime la session pour recommencer.")
                break
            try:
                with st.spinner(f"Analyse de {fichier.name}…"):
                    modele = charger_modele()
                    identifiant, nombre, pages_vides = indexer(
                        client, modele, st.session_state.session_id, fichier.name, contenu)
                st.session_state.documents[identifiant] = fichier.name
                st.success(f"{fichier.name} : {nombre} extraits.")
                if pages_vides:
                    st.warning(f"{pages_vides} page(s) sans texte ignorée(s). Les images ne sont pas lues.")
            except ValueError as erreur:
                st.error(str(erreur))
            except Exception:
                st.error("Analyse interrompue. Vérifie le fichier et la connexion, puis réessaie.")
    for nom in st.session_state.documents.values():
        st.text(nom)
    if st.button("Effacer la discussion"):
        st.session_state.messages = []
        st.rerun()
    if st.button("Supprimer mes documents et ma session"):
        try:
            supprimer_session(client, st.session_state.session_id)
            for cle in ("session_id", "documents", "messages", "dernier_appel"):
                del st.session_state[cle]
            st.rerun()
        except Exception:
            st.error("Suppression non confirmée. Réessaie avant de fermer cette page.")
    st.caption("Supprime tes documents avant de fermer la page. Un rechargement peut perdre l'accès à la session.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        afficher_sources(message.get("sources", []))

question = st.chat_input("Pose une question précise sur tes documents",
                         disabled=not st.session_state.documents)
if question:
    if len(question) > 1000:
        st.warning("Raccourcis la question : maximum 1 000 caractères.")
        st.stop()
    if time.monotonic() - st.session_state.dernier_appel < 5:
        st.info("Attends quelques secondes avant une nouvelle question.")
        st.stop()
    st.session_state.dernier_appel = time.monotonic()
    # Les deux derniers échanges aident à interpréter les questions de suivi.
    historique = [{"role": m["role"], "content": m["content"][:1200]}
                  for m in st.session_state.messages[-4:]]
    anciennes_questions = [m["content"] for m in historique if m["role"] == "user"]
    requete = "\n".join(anciennes_questions[-1:] + [question])
    try:
        with st.spinner("Recherche dans tes documents…"):
            modele = charger_modele()
            sources = rechercher(client, modele, st.session_state.session_id,
                                 st.session_state.documents, requete)
            reponse = generer(parametre("GROQ_API_KEY"),
                             parametre("GROQ_MODEL", "openai/gpt-oss-20b"),
                             question, sources, historique)
    except RateLimitError:
        st.warning("Quota Groq atteint. Réessaie plus tard ; le délai dépend du quota dépassé.")
        st.stop()
    except AuthenticationError:
        st.error("La clé Groq est refusée. Vérifie sa configuration.")
        st.stop()
    except Exception:
        st.error("Réponse indisponible. Vérifie la connexion et l'accès au modèle GROQ_MODEL.")
        st.stop()
    st.session_state.messages.extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": reponse, "sources": sources},
    ])
    st.rerun()
