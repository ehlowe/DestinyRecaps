import asyncio

from . import data_processing
from core import utils



async def recap_segment(category, texts, model_name=utils.ModelNameEnum.gpt_4o_mini):
    if texts==[]:
        return ""
    
    system_prompt="""You are a stream segment recapper with an active voice. The user will tell you the supposed category of the content and the text of the content. Your job is to make a 50 word recap of what was said in precise detail. 
    
Do not use any fluff, your response does not need to be gramatically correct. Start with what generally happened and then go into specifics. """
    user_prompt="Category: "+category+"\n\nText:"
    for text in texts:
        user_prompt+=text+"\n"
    print("USER PROMPT LEN: ", len(user_prompt))
    full_prompt=[{"role":"system", "content":system_prompt}, {"role":"user", "content":user_prompt}]
    response, cost = await utils.async_response_handler(full_prompt, model_name)
    return response, cost
    

async def recap_segments(plot_object: data_processing.PlotObject, model_name=utils.ModelNameEnum.gpt_4o_mini):
    # Produce the segment recaps
    tasks=[]
    for i in range(len(plot_object.segments)):
        tasks.append(recap_segment(plot_object.segments[i].category, plot_object.segments[i].texts, model_name))
    responses_and_costs=await asyncio.gather(*tasks)

    # load the responses into the plot object
    cost=0
    for i in range(len(plot_object.segments)):
        plot_object.segments[i].recap=responses_and_costs[i][0]
        cost+=responses_and_costs[i][1]

    return plot_object, cost



async def recap_abstraction(category, segment_recaps, model_name):
    if segment_recaps==[]:
        return ""

    system_prompt="""You are a stream segment recapper with an active voice. The user will tell you the supposed category of the content and the recaps of the contents. Your job is to make a 50 word recap of what was said, specifically in relation to the category, in precise detail. 
    
Do not use any fluff, your response does not need to be gramatically correct. Start with what generally happened and then go into specifics. """
    user_prompt="Category: "+category+"\n\nText:"
    for recap in segment_recaps:
        user_prompt+=recap+"\n"
    full_prompt=[{"role":"system", "content":system_prompt}, {"role":"user", "content":user_prompt}]
    response, cost = await utils.async_response_handler(full_prompt, model_name)
    return response, cost

async def recap_abstractions(plot_object: data_processing.PlotObject, model_name=utils.ModelNameEnum.gpt_4o_mini):
    # Produce the abstraction recaps
    tasks=[]
    for key in list(plot_object.abstractions.keys()):
        # get all the segments with category == key
        temp_segment_recaps=[]
        for segment in plot_object.segments:
            if segment.category==key:
                temp_segment_recaps.append(segment.recap)

        tasks.append(recap_abstraction(key, temp_segment_recaps, model_name))
    responses_and_costs=await asyncio.gather(*tasks)

    # load the responses into the plot object
    cost=0
    for i, key in enumerate(list(plot_object.abstractions.keys())):
        plot_object.abstractions[key].recap=responses_and_costs[i][0]
        cost+=responses_and_costs[i][1]


    return plot_object, cost





async def annotate_start(stream_recap_data, category, contextualized_start_index, text=None):
    if not text:
        return None, 0

    system_prompt="""You define the start point of a segment, the segment has text and a category. You must say where the starts by giving the text match of the starting point sufficient such that they can be searched programatically.
    
Do not say anthing else and you must produce your response as an exact text match, that may mean that you do not adhere to gramatical rules and even in the case of explicit content you must repeat the text exactly.

It is important that when you are starting where the start point is that it is the start point of the category given to you and not the start point of the text unless the text is the start point of the category. 

REMEBER YOU MUST GIVE THE TEXT NOT THE CATEGORY BUT THE TEXT WHEN IT STARTS TO FIT THE CATEGORY. ALWAYS DO YOUR BEST TO GUESS A REASONABLE START POINT AND THEN SAY ENOUGH SO THAT THE START POINT CAN BE SEARCHED."""

    full_prompt=[{"role":"system", "content":system_prompt}, {"role":"user", "content":"Category: "+category+"\n\nText: "+text}]
    response, cost = await utils.async_response_handler(full_prompt, utils.ModelNameEnum.gpt_4o_mini)
    # print(response)
    real_start_character_index=stream_recap_data.transcript[contextualized_start_index:].find(response)
    if real_start_character_index>=0:
        real_start_character_index+=contextualized_start_index

    return real_start_character_index, cost, response
    


async def exact_start_annotations(video_id, stream_recap_data, plot_object: data_processing.PlotObject):
    context_offset=500

    tasks=[]
    for i in range(len(plot_object.segments)):
        task_text=None
        if plot_object.segments[i].texts!=[]:
            start_index=stream_recap_data.transcript.find(plot_object.segments[i].texts[0])
            contextualized_start_index=start_index-context_offset
            if (contextualized_start_index)<0:
                contextualized_start_index=0
            task_text=stream_recap_data.transcript[contextualized_start_index:start_index+len(plot_object.segments[i].texts[0])]
        tasks.append(annotate_start(stream_recap_data, plot_object.segments[i].category, contextualized_start_index, task_text))
    start_counts_and_costs=await asyncio.gather(*tasks)




    # conver the character indexes to start times
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(stream_recap_data.linked_transcript, 'html.parser')
    soup_list=soup.find_all('a')

    # Setup list for getting the time at each character count
    transcript_soup_character_counter=[]
    character_count_total=0
    temp_time=0
    for link in soup_list:
        link_text = link.get_text()
        character_count_total+=len(link_text)

        if link.get('href') is not None:
            temp_time=int(link['href'].split("t=")[-1].split("s")[0])

        transcript_soup_character_counter.append([character_count_total, temp_time])


    # load the responses into the plot object
    cost=0
    counters=[0,0]
    for i, (start_character_count, temp_cost, response) in enumerate(start_counts_and_costs):
        # Get the real start time
        if start_character_count>=0:
            real_start_time=find_nearest_time_at_character_count(transcript_soup_character_counter, start_character_count)
            counters[1]+=1
        else:
            real_start_time=plot_object.segments[i].start_time

        counters[0]+=1

        # Place the real start as the seconds in the href
        plot_object.segments[i].href="https://youtu.be/"+video_id+"?t="+str(int(real_start_time))

        cost+=temp_cost

    print("Counters: ", counters)

    return plot_object, cost




def find_nearest_time_at_character_count(transcript_soup_character_counter, character_count):
    # middle sort the list to get to the closest character count quickly
    l=0
    r=len(transcript_soup_character_counter)-1

    while l<r:
        m=(l+r)//2
        if transcript_soup_character_counter[m][0]<character_count:
            l=m+1
        else:
            r=m

    if l==0:
        return transcript_soup_character_counter[l][1]
    else:
        if abs(transcript_soup_character_counter[l][0]-character_count)<abs(transcript_soup_character_counter[l-1][0]-character_count):
            return transcript_soup_character_counter[l][1]
        else:
            return transcript_soup_character_counter[l-1][1]
        