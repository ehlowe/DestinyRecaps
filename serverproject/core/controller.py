import time
import os
from dataclasses import dataclass, asdict, is_dataclass
from destinyapp.models import StreamRecapData, FastRecapData
import asyncio
from asgiref.sync import sync_to_async, async_to_sync
import traceback
from django.http import StreamingHttpResponse, JsonResponse
import json

# from .core import services
# from .core import utils

from . import utils
from . import services

# from destinyapp.




# Auto Recap Gen Controller
class auto_recap_controller:

    @classmethod
    async def run(self):
        print("Starting Auto Recap Controller")
        video_ids=await self.get_video_ids()
        print(video_ids)

        if video_ids!=[]:
            discord_message_video_ids=await self.generate_all(video_ids)

            if discord_message_video_ids!=[]:
                await self.send_discord_messages(discord_message_video_ids)

        # print("Writing to master vectordb")
        # await services.vector_db_and_text_chunks.AllRecapsVectorDB.write_master_all()

        #await services.timeline_plot.weekly_controller.attempt_weekly_recap()

    @classmethod
    async def manual_run(self, video_ids):
        print("Starting Manual Auto Recap Controller for video ids: ", video_ids)

        if video_ids!=[]:
            discord_message_video_ids=await self.generate_all(video_ids)

            if discord_message_video_ids!=[]:
                await self.send_discord_messages(discord_message_video_ids)
        await services.vector_db_and_text_chunks.AllRecapsVectorDB.write_master_all()






    async def get_video_ids():
        # Ensure that the bot can run
        bot_run_bool=await services.bot_run_check()
        if not bot_run_bool:
            print("Bot cannot run at this time, it has not been long enough since the last run.")
            return []

        # Get video ids to run on
        video_ids=await services.web_view_recent_stream_ids()

        return video_ids

        

    @classmethod
    async def generate_all(self, video_ids):
        # video_id_test="Gej2eHRwlM0"
        # video_id_test="kDsuggCYdyE"
        # video_id_test="SPaeA8shXFg"

        video_id_test=None
        # video_id_test="Krhk1FmL7b0"
        # video_id_test="-JNo1S9EDXI"
        video_id_test="nDINo-QO88Y"
        video_id_test="-EUyVSZEuIc"

        if os.environ.get("discord_channel")=="recaps":
            video_id_test=None

        print("Video Id Test:", video_id_test)


        discord_message_video_ids=[]

        # Loop through video ids
        for video_id in video_ids:

            # override with video_id_test
            if video_id_test!=None:
                video_id=video_id_test

            # Check run conditions for video_id
            test_stream_recap_data=await utils.get_recap_data(video_id)
            live_bool=await services.get_live_status(video_id)

            # wait a few seconds 
            asyncio.sleep(5)

            # override live_bool to test
            if video_id_test!=None:
                live_bool=False

            # Must not have existing data and must not be live
            if (not live_bool) and (test_stream_recap_data==None):
                generated_video_id=await self.generate(video_id)
                if generated_video_id!=None:
                    discord_message_video_ids.append(generated_video_id)
            else:
                if test_stream_recap_data!=None:
                    print("Stream Recap Data already exists for: ", video_id)
                if live_bool:
                    print("Stream is live for: ", video_id)

        return discord_message_video_ids
    
    @classmethod
    async def generate(self, video_id):
        try:
            print("STARTING RECAP GENERATION FOR: ", video_id)
            # Get all transcript data
            # transcript, linked_transcript, raw_transcript_data=await self.produce_transcript_data(video_id)


            # Get all recap data
            await services.video_and_transcript.download_youtube_transcript(video_id)
            raw_transcript_data=await services.video_and_transcript.read_and_process_youtube_transcript(video_id)
            transcript, linked_transcript=services.fast.transcript_processing.process_yt_transcript(raw_transcript_data, video_id)
            vectordb, text_chunks, segments_and_summaries, finalized_recap, video_characteristics=await self.produce_recap_data(video_id, transcript)

            # Save the data
            await self.save_data(video_id, video_characteristics, raw_transcript_data, transcript, linked_transcript, text_chunks, segments_and_summaries, finalized_recap)
            print("Saved Recap Data prior to plot generation")

            try: 
                await StreamPlotController.run(video_id)
            except Exception as e: 
                print("Error in auto_recap_controller.generate_all for plot generation: ", e)
                traceback.print_exc()

            # add the video id to the list of video ids to send discord messages for
            return video_id
        except Exception as e:
            print("Error in auto_recap_controller.generate: ", e)
            traceback.print_exc()
            return None


    # Produce Transcript Data
    async def produce_transcript_data(video_id):
        # Download video
        await services.download_video(video_id)

        # Generate Assembly Transcript
        raw_transcript_data = await services.generate_assembly_transcript()

        # Process Raw Transcript
        transcript, linked_transcript = await services.process_raw_transcript(raw_transcript_data, video_id)

        return transcript, linked_transcript, raw_transcript_data
    

    # Generate Everything Needed for Recap
    async def produce_recap_data(video_id, transcript):
        # Generate Vector DB and Text Chunks
        vectordb, text_chunks = await services.VectorDbAndTextChunksGenerator.generate_basic_vectordb_and_chunks(video_id, transcript)
        print("Vector DB and Text Chunks Generated")

        # Generate Summarized Segments
        segments_and_summaries = await services.SummarizedSegmentGenerator.generate_summarized_segments(transcript)

        # Generate Recap
        recap = await services.RecapGenerator.generate_recap(segments_and_summaries)

        # Generate Recap Hook
        recap_hook=await services.RecapGenerator.generate_recap_hook(recap)

        # Format and finalize the recap
        finalized_recap=recap_hook+"\n"+recap+"\n\nDISCLAIMER: This is all AI generated and there are frequent errors."

        # Get stream title
        video_characteristics=await services.get_video_characteristics(video_id)

        return vectordb, text_chunks, segments_and_summaries, finalized_recap, video_characteristics
    
    # Save the data
    async def save_data(video_id, video_characteristics, raw_transcript_data, transcript, linked_transcript, text_chunks, segments_and_summaries, finalized_recap):
        stream_recap_data=StreamRecapData(video_id=video_id, video_characteristics=video_characteristics, raw_transcript_data=raw_transcript_data, transcript=transcript, linked_transcript=linked_transcript, text_chunks=text_chunks, summarized_chunks=segments_and_summaries, recap=finalized_recap)

        await sync_to_async(stream_recap_data.save)()


    async def send_discord_messages(discord_message_video_ids):
        #oldest to newest
        discord_message_video_ids.reverse()

        # Compile data for recap messages
        discord_recaps_to_send=await services.DiscordMessageHandler.compile_discord_messages(discord_message_video_ids)

        # Send recaps to discord
        await services.DiscordMessageHandler.send_discord_recaps(discord_recaps_to_send)


    # Generate Stream Plot
    async def generate_stream_plot(video_id):
        await StreamPlotController.run(video_id)





def str_to_bool(value):
    return value.lower() in ('true', '1', 't', 'y', 'yes')


class update_controller:
    @classmethod
    async def update(self):
        stream_recaps_limited = await utils.get_all_recaps_fast()

        override=str_to_bool(os.environ.get("update_over_bool", "false"))

        # Update the latest plots
        #await self.update_latest_plots(stream_recaps_limited, override=override) 

        # # Update the transcript processing
        # await self.update_transcript_processing(stream_recaps_limited)

        # # update the video_characteristics
        # await self.update_video_characteristics(stream_recaps_limited, override=override)

        print("Starting update_weekly_recaps")
        await self.update_weekly_recaps()
    
    async def update_latest_plots(stream_recaps_limited, update_range=int(os.environ.get("update_range")), override=False):
        # Get the video ids
        video_ids=[]
        for stream_recap in stream_recaps_limited:
            video_ids.append(stream_recap["video_id"])
        print("Video Ids to potentially update: ", video_ids)

        for video_id in video_ids[0:update_range]:
            # Update the raw transcript processing
            stream_recap_data=await utils.get_recap_data(video_id)
            transcript, linkted_transcript = await services.process_raw_transcript(stream_recap_data.raw_transcript_data, video_id)
            stream_recap_data.transcript=transcript
            stream_recap_data.linked_transcript=linkted_transcript
            await sync_to_async(stream_recap_data.save)()

            # get stream recap data
            stream_recap=await utils.get_recap_data(video_id)

            # Generate the stream plot
            if override or len(stream_recap.plot_image)<100:
                await StreamPlotController.run(video_id)
            else:
                print("Plot already exists for: ", video_id)


    async def update_transcript_processing(stream_recaps_limited, update_range=int(os.environ.get("update_range"))):
        # Process Raw Transcript
        video_ids=[]
        for stream_recap in stream_recaps_limited:
            video_ids.append(stream_recap["video_id"])

        for video_id in video_ids[0:update_range]:
            try:
                stream_recap_data=await utils.get_recap_data(video_id)
                transcript, linked_transcript = await services.process_raw_transcript(stream_recap_data.raw_transcript_data, video_id)
                stream_recap_data.transcript=transcript
                stream_recap_data.linked_transcript=linked_transcript
                await sync_to_async(stream_recap_data.save)()
            except Exception as e:
                print("Error in update_controller.update_process for transcript processing: ", e)
                traceback.print_exc()

    async def update_video_characteristics(stream_recaps_limited, update_range=int(os.environ.get("update_range")), override=False):
        # Get the video ids
        video_ids=[]
        for stream_recap in stream_recaps_limited:
            video_ids.append(stream_recap["video_id"])
        print("Video Ids to potentially update: ", video_ids)

        for video_id in video_ids[0:update_range]:
            try:
                # get stream recap data
                stream_recap=await utils.get_recap_data(video_id)

                # Get stream title
                if override:
                    video_characteristics=await services.get_video_characteristics(video_id)
                else:
                    if video_characteristics.get("title", "")=="":
                        video_characteristics=await services.get_video_characteristics(video_id)
                    else:
                        print("Video Characteristics already exist for: ", video_id)
                        continue

                # Save the data
                stream_recap.video_characteristics=video_characteristics
                await sync_to_async(stream_recap.save)()

                await asyncio.sleep(5)
            except Exception as e:
                print("Error in update_controller.update_video_characteristics: ", e)
                traceback.print_exc()


    async def update_weekly_recaps():
        await services.timeline_plot.weekly_controller.update_weekly_recaps()













import json
from youtube_transcript_api import YouTubeTranscriptApi
from dataclasses import asdict



class StreamPlotController:
    @classmethod
    async def run(self, video_id):
        cost=0
        stream_recap_data=await utils.get_recap_data(video_id)

        annotated_results, major_topics, minor_topics, temp_cost = await services.stream_plot.generate_data(stream_recap_data)
        cost+=temp_cost

        plot_object, annotated_results, plot_segments, category_locations = await services.stream_plot.process_data(stream_recap_data,  annotated_results, major_topics, minor_topics, video_id)

        plot_object, temp_cost=await services.stream_plot.annotate_extra(video_id, stream_recap_data, plot_object)  
        cost+=temp_cost

        # save plot object to the database
        stream_recap_data.plot_object=asdict(plot_object)
        await sync_to_async(stream_recap_data.save)()

        # grab image and save it
        await services.stream_plot.visit_until_image_saved(video_id)

        print("Plotting Process Cost: ", cost)




class FastPlotController:
    @classmethod
    async def return_plot(self, video_id):
        # if a plot exists, return it, otherwise generate it
        fast_recap_filter=await sync_to_async(FastRecapData.objects.filter)(video_id=video_id)
        fast_recap_data=await sync_to_async(fast_recap_filter.first)()
        if fast_recap_data!=None:
            return fast_recap_data
        else:
            fast_recap_data=FastRecapData(video_id=video_id)
            await sync_to_async(fast_recap_data.save)()

            try:
                plot_object = await self.run(video_id)
                fast_recap_data.plot_object=asdict(plot_object)
                plot_parameters=fast_recap_data.plot_object["plot_parameters"]
                await sync_to_async(fast_recap_data.save)()
            except Exception as e:
                print("Error in FastPlotController.return_plot: ", e)
                traceback.print_exc()
                return None
            return fast_recap_data

    @classmethod
    async def run(self, video_id):
        cost=0

        # # For testing
        # if not os.path.exists("raw_transcript.json"):
        #     raw_transcript=YouTubeTranscriptApi.get_transcript(video_id)
        #     with open("raw_transcript.json", "w") as f:
        #         json.dump(raw_transcript, f)
        # else:
        #     with open("raw_transcript.json", "r") as f:
        #         raw_transcript=json.load(f)


        times=[]
        t_start=time.time()
        times.append({"start":t_start})

        # Get Transcript
        await services.video_and_transcript.download_youtube_transcript(video_id)
        raw_transcript=await services.video_and_transcript.read_and_process_youtube_transcript(video_id)
        transcript, linked_transcript=services.fast.transcript_processing.process_yt_transcript(raw_transcript, video_id)
        # if True:
        #     print("IN TESTING MODE")
        #     test_folder='destinyapp/working_folder/test_transcript/transcript.json'
        #     with open(test_folder, 'r') as f:
        #         data=json.load(f)
        #     transcript, linked_transcript=tuple(data.values())
        #     print(transcript, linked_transcript)
        #     print("LOADED TEST TRANSCRIPT")





        if (len(utils.api_requests.enc.encode(transcript)))>(120*1000):
            raise Exception("Transcript too long")
        t_transcript=time.time()
        times.append({"transcript":t_transcript})


        # Make Topics
        topics_str, major_topics, temp_cost=await services.fast.data_gen.create_topics(transcript)
        cost+=temp_cost
        t_topics=time.time()
        times.append({"topics":t_topics})

        # Annotate Text batches
        text_chunks_no_overlap = await services.stream_plot.data_gen.create_text_chunks(transcript, 0)
        text_chunk_batches = await services.stream_plot.data_gen.generate_text_chunk_batches(text_chunks_no_overlap)
        responses, annotated_results, temp_cost = await services.fast.data_gen.annotate_all_batches(text_chunk_batches, topics_str)
        cost+=temp_cost
        t_annotations=time.time()
        times.append({"annotations":t_annotations})

                
        # Create Plot
        annotated_segments, category_locations = await services.stream_plot.data_processing.create_segments(linked_transcript, annotated_results, major_topics, transcript)
        plot_segments=await services.stream_plot.data_processing.annotated_to_plot_segments(annotated_segments)
        plot_object=await services.stream_plot.data_processing.create_plot_object(plot_segments, category_locations, video_id)
        t_plot_object=time.time()
        times.append({"plot_object":t_plot_object})

        # Annotate Extra
        plot_object, temp_cost=await services.stream_plot.extra_annotations.recap_segments(plot_object)
        cost+=temp_cost
        t_recap_segments=time.time()
        times.append({"recap_segments":t_recap_segments})
        plot_object, temp_cost=await services.stream_plot.extra_annotations.recap_abstractions(plot_object)
        cost+=temp_cost
        t_recap_abstractions=time.time()
        times.append({"recap_abstractions":t_recap_abstractions})
        

        print("Created Fast Recap for video_id: ", video_id, " with cost: ", cost)

        for i in range(1,len(times)):
            print(list(times[i].keys())[0], ": ", list(times[i].values())[0]-list(times[i-1].values())[0])

        print("Total Time: ", list(times[-1].values())[0]-list(times[0].values())[0])

        # return plot
        return plot_object
    
    @classmethod
    async def generation_step(self, transcript, linked_transcript, video_id):
        times=[]
        cost=0


        # model_select=utils.ModelNameEnum.llama_3_2_90B_vision_preview
        model_select=utils.ModelNameEnum.llama_3_70b_8192


        # Make Topics
        # topics_str, major_topics, temp_cost=await services.fast.data_gen.create_topics(transcript, utils.ModelNameEnum.llama_3_1_70B_versatile)
        topics_str, major_topics, temp_cost=await services.fast.data_gen.create_topics_split(transcript, utils.ModelNameEnum.llama_3_2_90B_vision_preview)
        cost+=temp_cost
        t_topics=time.time()
        times.append({"topics":t_topics})
        print("Created Topics")

        # Annotate Text batches
        text_chunks_no_overlap = await services.stream_plot.data_gen.create_text_chunks(transcript, 0)
        text_chunk_batches = await services.stream_plot.data_gen.generate_text_chunk_batches(text_chunks_no_overlap)
        responses, annotated_results, temp_cost = await services.fast.data_gen.annotate_all_batches(text_chunk_batches, topics_str, model_select)
        cost+=temp_cost
        t_annotations=time.time()
        times.append({"annotations":t_annotations})
        print("Annotated Text Batches")

                
        # Create Plot
        annotated_segments, category_locations = await services.stream_plot.data_processing.create_segments(linked_transcript, annotated_results, major_topics, transcript)
        plot_segments=await services.stream_plot.data_processing.annotated_to_plot_segments(annotated_segments)
        plot_object=await services.stream_plot.data_processing.create_plot_object(plot_segments, category_locations, video_id)
        t_plot_object=time.time()
        times.append({"plot_object":t_plot_object})
        print("Created Plot Object")

        # Annotate Extra
        model_select=utils.ModelNameEnum.llama_3_2_90B_vision_preview
        # plot_object, temp_cost=await services.stream_plot.extra_annotations.recap_segments(plot_object, model_select)
        # cost+=temp_cost
        t_recap_segments=time.time()+0.0001
        times.append({"recap_segments":t_recap_segments})

        model_select=utils.ModelNameEnum.llama_3_70b_8192
        plot_object, temp_cost=await services.stream_plot.extra_annotations.recap_abstractions(plot_object, model_select)
        cost+=temp_cost
        t_recap_abstractions=time.time()
        times.append({"recap_abstractions":t_recap_abstractions})
        

        print("Created Fast Recap for video_id: ", video_id, " with cost: ", cost)

        for i in range(1,len(times)):
            print(list(times[i].keys())[0], ": ", list(times[i].values())[0]-list(times[i-1].values())[0])

        print("Total Time: ", list(times[-1].values())[0]-list(times[0].values())[0])

        return plot_object





















# class StreamConsumer(AsyncHttpConsumer):
#     async def handle(self, body):
#         await self.send_headers(headers=[
#             (b"Cache-Control", b"no-cache"),
#             (b"Content-Type", b"text/event-stream"),
#             (b"Transfer-Encoding", b"chunked"),
#         ])

#         for i in range(15):
#             data = {"progress": i+1, "data": f"Some data {i+1}"}
#             await self.send_body(f"data: {json.dumps(data)}\n\n".encode("utf-8"), more_body=True)
#             await asyncio.sleep(1)  # Simulate async processing time

#         await self.send_body(b"", more_body=False)