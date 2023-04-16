
from base_class import Embedding_Model
import pickle
import torch
from transformers import AutoTokenizer, AutoModel





#class HuggingfaceSentenceTransformerModel(Embedding_Model):
#    EMBEDDING_MODEL = "distiluse-base-multilingual-cased-v2"
#
#    def __init__(self, model_name=EMBEDDING_MODEL) -> None:
#        super().__init__(model_name)
#        
#        #self.model = SentenceTransformer(model_name)
#
#    def __call__(self, text) -> None:
#        return self.model.encode(text)






class SentenceTransformerMsmarcoModel(object):
    

    def __init__(self, model_name="sentence-transformers/msmarco-distilbert-base-v4"):

        self.model = AutoModel.from_pretrained(model_name).to("cuda")#'sentence-transformers/msmarco-distilbert-base-v4'
        print("embedding model device_map")
        print(self.model.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def __call__(self, text):
        # Tokenize sentences
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to("cuda")

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling. In this case, max pooling.
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings


    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)




# class OpenAIEmbeddingModel(Embedding_Model):
#     # constants
#     EMBEDDING_MODEL = "text-embedding-ada-002"
#     # establish a cache of embeddings to avoid recomputing
#     # cache is a dict of tuples (text, model) -> embedding, saved as a pickle file

#     def __init__(self, model_name=EMBEDDING_MODEL) -> None:
#         super().__init__(model_name)
#         self.model_name = model_name

#     # define a function to retrieve embeddings from the cache if present, and otherwise request via the API
#     def embedding_from_string(self,
#                               string: str,
#                               ) -> list:
#         """Return embedding of given string, using a cache to avoid recomputing."""
#         model = self.model_name
#         if (string, model) not in self.embedding_cache.keys():
#             self.embedding_cache[(string, model)] = get_embedding(
#                 string, model)
#             with open(self.embedding_cache_path, "wb") as embedding_cache_file:
#                 pickle.dump(self.embedding_cache, embedding_cache_file)
#         return self.embedding_cache[(string, model)]

#     def __call__(self, text) -> None:
#         return self.embedding_from_string(text)
