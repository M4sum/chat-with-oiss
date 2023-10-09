from utils.utils import *
client = QdrantClient(
            url=os.getenv("QDRANT_HOST"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
create_qdrant_collection(client, "Northwestern University")
raw_text = get_documents(url)
text_chunks = get_text_chunks(raw_text)
vectorstore = get_qdrant_vectorstore(client, new_collection_name)
vectorstore.add_texts(text_chunks)