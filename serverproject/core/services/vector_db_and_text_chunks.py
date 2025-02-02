import os
import asyncio
import json
import numpy as np
import faiss


from core import utils


class VectorDbAndTextChunksGenerator:
    model_name=utils.ModelNameEnum.claude_3_haiku
    text_chunk_summary_batch_prompt="""You will be given a transcript that is broken down into segments and you must give a overview and a 1-2 sentence summary for each text segment, label the segments with a number like (1.) and then give the overview and summary. No more than 200 characters per segment. The segments are from a youtube streamer named Destiny. The overview should be a general idea that you would present to someone as a topic. The summary should have more detail, like 2 sentences.

The overview is meant to be used by a vector embedding model to allow users to search for the spot something happened in from the transcript. People are going to search topics from a recap provided to them which is the need for this.

Here is an example of a segment and the overview and summary:
Text Segment: This says someone was assaulted with a cigarette and had burns on their body. They were attacked after getting into an argument with a customer. What is this, Jesus. Why are people so unhinged. Is there any proof of this story or is this just some random person saying something. 
Overview: Discussion of alleged cigarette burns and injuries in an unverified story.
Summary: A person was allegedly assaulted with a cigarette and had burns on their body after an argument with a customer. The story is unverified and the speaker questions the validity of the story.


    
Here is an example of the formatting:
"
(1.) Overview:
Summary:

(2.)Overview:
Summary:
...

When generating the points it should be from the context of the text segments provided by the user, generate for every single segment in the context provided.
"""

    async def text_chunk_summary_batch(text_batch, sys_prompt=text_chunk_summary_batch_prompt, model_name=model_name):
        prompt=[{"role":"system","content":sys_prompt}, {"role": "user", "content": "Here are the text segments: "+text_batch}]
        response_str, cost=await utils.async_response_handler(
            prompt=prompt,
            model_name=model_name,
        )


        return response_str, cost
    
    # Generate vectordb and chunks for assemblyai transcript
    async def generate_basic_vectordb_and_chunks(video_id, transcript):
        """
        Saves a vectordb to the 'video_id'.index file 
        
        Creates text chunks tied to the vectordb 
        
        Returns dictionary of 'vector_db' and 'text_chunks'"""
        # Make chunks
        chunk_size=1000
        overlap=500
        text_chunks=[transcript[i:i+chunk_size] for i in range(0, len(transcript), (chunk_size-overlap))]
        print("Number of chunks: ",len(text_chunks))

        # make vector db
        async def make_vector_db_fast(openai_client, text_chunks):
            # Async function to fetch embeddings
            async def generate_embeddings_async(text_chunks, model):
                model="text-embedding-3-large"
                async def fetch_embedding(chunk):
                    # Simulate an async call to the embeddings API
                    #return await asyncio.to_thread(openai_client.embeddings.create, input=chunk, model=model)

                    fails=0
                    while fails<5:
                        try:
                            return await openai_client.embeddings.create(input=chunk, model=model)
                        except Exception as e:
                            fails+=1
                            print("Emedding Fail Retrying:",e)
                            await asyncio.sleep(10+(fails*2))
                    return None
                    

                responses = await asyncio.gather(*(fetch_embedding(chunk) for chunk in text_chunks))
                embeddings = [response.data[0].embedding for response in responses]
                return np.array(embeddings)

            # Generate embeddings
            model="text-embedding-3-large"
            embeddings=await generate_embeddings_async(text_chunks, model)
            print("Finished generating embeddings")

            # Make vector db
            vector_db=faiss.IndexFlatL2(embeddings.shape[1])
            vector_db.add(np.array(embeddings))
            print("Finished making vector db")

            return vector_db
        
        # Make vectordb
        vector_db= await make_vector_db_fast(utils.async_openai_client, text_chunks)

        # Save vector db
        vectordb_folder_path="destinyapp/working_folder/vectordbs/"
        if not os.path.isdir(vectordb_folder_path):
            os.mkdir(vectordb_folder_path)
        faiss.write_index(vector_db, vectordb_folder_path+video_id+".index")
        print("Saved vector db to:", vectordb_folder_path+video_id+".index")
        
        # return vector db and text chunks
        return vector_db, text_chunks
    


class AllRecapsVectorDB:

    # @classmethod
    # async def load_master_vector_db(self, video_id):
    #     # Load the master vector db
    #     vectordb_folder_path="destinyapp/working_folder/vectordbs/"
    #     vector_db_file_name="master.index"
    #     if not os.path.isdir(vectordb_folder_path):
    #         os.mkdir(vectordb_folder_path)
    #     if not os.path.isfile(vectordb_folder_path+vector_db_file_name):
    #         print("Master vector db not found, creating a new one")
    #         test_embedding=await self.fetch_embedding("test")
    #         if test_embedding:
    #             vector_db=faiss.IndexFlatL2(test_embedding.shape[1])

    #         faiss.write_index(vector_db, vectordb_folder_path+vector_db_file_name)

    #     vector_db=faiss.read_index(vectordb_folder_path+"master.index")
    #     return vector_db

    # def add_video_to_master_db(video_id, vector_db):
    #     pass

    @classmethod
    async def write_master_all(self):
        all_recaps=await utils.get_all_recaps_fast()

        segment_length=240
        overlap_length=40


        recaps_to_write=[]
        ids_mapping={}
        counter=0
        for recap in all_recaps:
            if recap.get("recap", None):
                raw_recap_str=recap["recap"]
                recap_segments=[]
                for i in range(0, len(raw_recap_str), (segment_length-overlap_length)):
                    ids_mapping[counter]=recap["video_id"]
                    counter+=1
                    recap_segments.append(raw_recap_str[i:i+segment_length])
                
                recaps_to_write+=recap_segments

        # write ids_mapping to json
        with open("destinyapp/working_folder/vectordbs/master_all.json", "w") as f:
            json.dump(ids_mapping, f)

        vector_db=await self.make_vector_db_fast(recaps_to_write)

        # Save vector db
        vectordb_folder_path="destinyapp/working_folder/vectordbs/"
        if not os.path.isdir(vectordb_folder_path):
            os.mkdir(vectordb_folder_path)
        
        faiss.write_index(vector_db, vectordb_folder_path+"master_all.index")
        print("Saved master_all vector db to:", vectordb_folder_path+"master_all.index")
        return
    

    # make vector db
    @classmethod
    async def make_vector_db_fast(self, recaps):
        # Async function to fetch embeddings
        async def generate_embeddings_async(text_chunks):
            model="text-embedding-3-large"
            async def fetch_embedding(chunk):
                fails=0
                while fails<5:
                    try:
                        return await utils.async_openai_client.embeddings.create(input=chunk, model=model)
                    except Exception as e:
                        fails+=1
                        print("Emedding Fail Retrying:",e)
                        await asyncio.sleep(10+(fails*2))
                return None
                

            responses = await asyncio.gather(*(fetch_embedding(chunk) for chunk in text_chunks))
            embeddings = [response.data[0].embedding for response in responses]
            total_cost=0
            for response in responses:
                total_cost+=(0.00013*response.usage.total_tokens/1000)
            print("Total Cost of Embeddings for ALL bot db:", total_cost)
            return np.array(embeddings)

        # Generate embeddings
        embeddings=await generate_embeddings_async(recaps)
        print("Finished generating embeddings")

        # Make vector db
        vector_db=faiss.IndexFlatL2(embeddings.shape[1])
        vector_db.add(np.array(embeddings))
        print("Finished making vector db")

        return vector_db

    @classmethod
    async def fetch_embedding(self, text):

        # Simulate an async call to the embeddings API
        #return await asyncio.to_thread(openai_client.embeddings.create, input=chunk, model=model)
        model="text-embedding-3-large"

        fails=0
        while fails<5:
            try:
                return await utils.async_openai_client.embeddings.create(input=text, model=model)
            except Exception as e:
                fails+=1
                print("Emedding Fail Retrying:",e)
                await asyncio.sleep(10+(fails*2))
        return None