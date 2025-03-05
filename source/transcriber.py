import whisper
import os 
from pydub import AudioSegment
from pyannote.audio import Pipeline
# from concurrent.futures import ThreadPoolExecutor
# from scipy.io import wavfile
import multiprocessing
import torch
import spacy
import re
import subprocess
import shutil
import language_tool_python

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
    use_auth_token='...').to(device)
#Token is removed here. In case needed for usage, you can use installer .exe file

# Load the Whisper model
model = whisper.load_model("medium.en").to(device)


def transcribe_audio(file_path):
    # Transcribe the audio file
    result = model.transcribe(file_path)
    return result['text']

# Modified 'extract_audio_segment' function
def extract_audio_segment(file_path, start_time, end_time, batch_dir, new_file_flag):
    # Load the original audio file
    audio = AudioSegment.from_file(file_path)

    # Convert start and end times from seconds to milliseconds
    start_ms = start_time * 1000
    end_ms = end_time * 1000

    # Extract the audio segment
    audio_segment = audio[start_ms:end_ms]

    # Initialize static variable for file numbering
    if not hasattr(extract_audio_segment, "next_file_number") or new_file_flag:
        extract_audio_segment.next_file_number = 1
    else:
        extract_audio_segment.next_file_number += 1

    # Generate a unique filename for the audio segment
    segment_filename = f"audio_{extract_audio_segment.next_file_number}.wav"
    segment_path = os.path.join(batch_dir, segment_filename)

    # Export the audio segment
    audio_segment.export(segment_path, format="wav")

    return segment_path

# Assuming `transcribed_segments` is the list you obtained from `detect_speech_segments`
# and `file_path` is the path to the original audio file
# Call the function like this:
# export_transcriptions_to_txt(transcribed_segments, file_path)

# Load spaCy model
nlp = spacy.load("en_core_web_lg")

# Print the path to the spaCy model
#model_path = spacy.util.get_package_path("en_core_web_sm")
#print(f"spaCy model path: {model_path}")

def segment_text_with_spacy(text, max_length=42):
    # Process the text with spaCy
    doc = nlp(text)
    # Initialize variables to store the current line and the list of segmented lines
    current_line = ""
    segmented_lines = []
    for token in doc:
        # Check if adding the next token exceeds the max length
        if len(current_line + token.text) + 1 > max_length:
            # If so, add the current line to the list and start a new line
            segmented_lines.append(current_line)
            current_line = token.text
        else:
            # Otherwise, add the token to the current line
            current_line += " " + token.text if current_line else token.text
    # Add the last line to the list if it's not empty
    if current_line:
        segmented_lines.append(current_line)
    #return segmented_lines

    # Merge lines containing only punctuation marks with the previous line
    i = 1
    while i < len(segmented_lines):
        if len(segmented_lines[i]) == 1 and segmented_lines[i] in ".?!,":
            segmented_lines[i-1] += segmented_lines[i]
            del segmented_lines[i]
        else:
            i += 1

    # Remove extra spaces before punctuation marks
    corrected_lines = []
    for line in segmented_lines:
        corrected_line = line.replace(" 's", "'s")
        corrected_line = corrected_line.replace(" ,", ",")
        corrected_line = corrected_line.replace(" .", ".")
        corrected_line = corrected_line.replace(" ?", "?")
        corrected_line = corrected_line.replace(" !", "!")
        corrected_lines.append(corrected_line)

    return corrected_lines     



def format_time_srt(time_in_seconds):
    hours = int(time_in_seconds // 3600)
    minutes = int((time_in_seconds % 3600) // 60)
    seconds = int(time_in_seconds % 60)
    milliseconds = int((time_in_seconds - int(time_in_seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def add_start_time_to_srt(srt_file_path, start_time):
    with open(srt_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    new_lines = []
    time_pattern = re.compile(r'(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})')
    
    for line in lines:
        match = time_pattern.match(line)
        if match:
            start_hours, start_minutes, start_seconds, start_milliseconds, end_hours, end_minutes, end_seconds, end_milliseconds = map(int, match.groups())
            start_time_seconds = start_hours * 3600 + start_minutes * 60 + start_seconds + start_milliseconds / 1000 + start_time
            end_time_seconds = end_hours * 3600 + end_minutes * 60 + end_seconds + end_milliseconds / 1000 + start_time
            new_start_time = format_time_srt(start_time_seconds)
            new_end_time = format_time_srt(end_time_seconds)
            new_line = f"{new_start_time} --> {new_end_time}\n"
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    with open(srt_file_path, 'w', encoding='utf-8') as file:
        file.writelines(new_lines)
  

def format_time_srt(time_in_seconds):
        # Placeholder for the time formatting function
        hours, remainder = divmod(time_in_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}:{seconds:06.3f}".replace('.', ',')


# Function to convert SRT time format to seconds
def srt_time_to_seconds(srt_time):
    h, m, s, ms = map(int, re.split('[:,]', srt_time))
    return h * 3600 + m * 60 + s + ms / 1000
    
# Function to convert seconds to SRT time format
def seconds_to_srt_time(seconds):
    ms = int((seconds % 1) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f'{h:02}:{m:02}:{s:02},{ms:03}'

def check_overlapping_entries(srt_file_path):
    # Read the SRT file and parse the entries
    with open(srt_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regular expression to match SRT entries
    pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)\n\n', re.DOTALL)
    parsed_entries = pattern.findall(content)

    overlap_count = 0  # Counter for overlapping entries

    # Check for overlapping entries and adjust times
    for i in range(len(parsed_entries) - 1):
        start_i, end_i, text_i = parsed_entries[i][1], parsed_entries[i][2], parsed_entries[i][3]
        end_i_sec = srt_time_to_seconds(end_i)

        for j in range(i + 1, len(parsed_entries)):
            start_j, end_j, text_j = parsed_entries[j][1], parsed_entries[j][2], parsed_entries[j][3]
            start_j_sec = srt_time_to_seconds(start_j)

            # Check for overlap
            if end_i_sec > start_j_sec:
                overlap_count += 1  # Increment the counter
                # Adjust the end time of the overlapping entry
                new_end_i_sec = start_j_sec - 0.1
                parsed_entries[i] = (parsed_entries[i][0], start_i, seconds_to_srt_time(new_end_i_sec), text_i)
                # Print the pair of overlapped entries
                print(f'Entries {i + 1} and {j + 1} are overlapping')

    # Print the number of overlapping pairs
    print(f'Number of overlapping pairs: {overlap_count}')

    # Write the adjusted entries back to the SRT file
    with open(srt_file_path, 'w', encoding='utf-8') as file:
        for entry in parsed_entries:
            index, start_time, end_time, text = entry
            file.write(f'{index}\n')
            file.write(f'{start_time} --> {end_time}\n')
            file.write(f'{text}\n\n')


   
def parse_textgrid_to_entries(textgrid_content):
    """
    Parses the TextGrid content and extracts entries based on consecutive lines
    containing 'xmin', 'xmax', and 'text' keywords, stopping the process if "item [2]:"
    is found within 7 lines before the 'xmin' line.
    """
    lines = textgrid_content.splitlines()
    entries = []
    i = 0

    while i < len(lines) - 2:  # Iterate through lines, stopping at the third-to-last line
        # Check for "item [2]:" within the previous 7 lines
        if any("item [2]:" in line for line in lines[max(0, i-7):i]):
            break  # Stop processing if "item [2]:" is found

        if "xmin =" in lines[i] and "xmax =" in lines[i + 1] and "text =" in lines[i + 2]:
            # Extract values using regular expressions
            xmin = re.search(r'xmin = (.+)', lines[i]).group(1)
            xmax = re.search(r'xmax = (.+)', lines[i + 1]).group(1)
            text_match = re.search(r'text = "(.*?)"', lines[i + 2])
            if text_match:  # Ensure the text is not empty
                text = text_match.group(1).strip()
                if text:
                    # Add the extracted values to the entries list
                    entries.append([xmin, xmax, text, False])
            i += 3  # Move to the next set of lines after a successful match
        else:
            i += 1  # Move to the next line if the current line doesn't match

    return entries

enter=0

def postprocess_srt(srt_file_path):
    # Initialize the language tool
    tool = language_tool_python.LanguageTool('en-US')

    # Read the SRT file
    with open(srt_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    corrected_lines = []
    temp_line = ""
    for line in lines:
        # Check if the line is a subtitle entry (not a timestamp or index)
        if re.match(r'^\d+$', line.strip()) or re.match(r'^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$', line.strip()):
            corrected_lines.append(line)
        else:
            # Check if the line is fully uppercased
            if line.isupper():
                # Lowercase all words except the first letter of the entry
                line = line.lower().capitalize()

            # Use language tool to check and correct the line
            #matches = tool.check(line)

            #corrected_line = language_tool_python.utils.correct(line, matches)       

            # Copy the content of line to temp_line
            temp_line = line

            # Use language tool to check and correct the line
            matches = tool.check(line)
            corrected_line = language_tool_python.utils.correct(line, matches)

            # Double check with the corrected_line
            if temp_line and temp_line[0].islower() and corrected_line[0].isupper():
                corrected_line = corrected_line[0].lower() + corrected_line[1:]

            corrected_lines.append(corrected_line)    

    # Write the corrected lines back to the SRT file
    with open(srt_file_path, 'w', encoding='utf-8') as file:
        file.writelines(corrected_lines)

def convert_textgrid_to_srt(textgrid_file_path, srt_file_path):
    global enter #Declare enter as global
    # Construct the path to the corresponding TXT file in the final_segmented folder
    base_name = os.path.basename(textgrid_file_path)
    txt_file_name = base_name.replace('.TextGrid', '.txt')
    grouping_txt_file_path = os.path.join(os.path.dirname(textgrid_file_path), txt_file_name)

    # Read the .textgrid file
    with open(textgrid_file_path, 'r', encoding='utf-8') as file:
        textgrid_content = file.read()

    # Parse the TextGrid content, add a boolean to track usage, and ignore empty text entries
    entries = parse_textgrid_to_entries(textgrid_content)

    #enter+=1
    #if(enter==13):
    #    for entry in entries:
    #        print(entry)
        #enter=True

    # Read the grouping TXT file
    with open(grouping_txt_file_path, 'r', encoding='utf-8') as file:
        groups = file.readlines()

    # Initialize a word counter
    word_counter = 0
    
    # Write the SRT file
    with open(srt_file_path, 'w', encoding='utf-8') as srt_file:
        index = 1
        for group in groups:
            words = group.strip().split()
            if words:
                group_start_time, group_end_time = float('inf'), float('-inf')
                for word in words:
                    word_counter += 1  # Increment the word counter for each word
                    start_time, end_time = None, None # Initialize start and end times for the current word
                    if word_counter <= len(entries):
                        entry = entries[word_counter - 1]  # Access the entry using word_counter
                        start_time, end_time = entry[0], entry[1]
                        #print(f"{entry[0]} {entry[1]}\n")                    
                    
                    if start_time is not None:
                        group_start_time = min(group_start_time, float(start_time))
                    if end_time is not None:
                        group_end_time = max(group_end_time, float(end_time))

                if group_start_time != float('inf') and group_end_time != float('-inf'):
                    # Convert times to SRT format and write to file
                    srt_start_time = format_time_srt(group_start_time)
                    srt_end_time = format_time_srt(group_end_time)
                    srt_file.write(f"{index}\n{srt_start_time} --> {srt_end_time}\n{' '.join(words)}\n\n")
                    index += 1
    


# Modified 'fragment_and_align_transcript' function
def fragment_and_align_transcript(transcription, start_time, batch_dir, new_file_flag):
    # Initialize static variable for file numbering
    if not hasattr(fragment_and_align_transcript, "next_file_number") or new_file_flag:
        fragment_and_align_transcript.next_file_number = 1
    else:
        fragment_and_align_transcript.next_file_number += 1

    # Segment the transcription into sentences
    doc = nlp(transcription)
    sentences = [sent.text.strip() for sent in doc.sents]


    # Write sentences to a new file in the batch directory
    segment_filename = f"audio_{fragment_and_align_transcript.next_file_number}.txt"
    segment_file_path = os.path.join(batch_dir, segment_filename)
    with open(segment_file_path, 'w', encoding='utf-8') as file:
        for sentence in sentences:
            # Segment each sentence if it's longer than 42 characters using spaCy
            segmented_sentences = segment_text_with_spacy(sentence, 42)
            for segment in segmented_sentences:
                file.write(segment + '\n')

    # # Write start_time to mfa_time_log.txt in the run folder
    time_log_path = os.path.join(run_folder_dir, "mfa_time_log.txt")
    with open(time_log_path, "a") as log_file:
        log_file.write(f"{start_time}\n")
    

# Define run_folder_dir as the "run" folder in the current directory
run_folder_dir = os.path.join(os.getcwd(), "run")

# Define batch_text_dir as the "batch_text" folder inside the run folder
batch_text_dir = os.path.join(run_folder_dir, "batch_text")
os.makedirs(batch_text_dir, exist_ok=True)
            
#Define audio_path as the path to the audio file
audio_path = os.path.join(run_folder_dir, "temp_converted_audio.wav")

# Function to process each batch
def process_batch(batch_segments, batch_number):
    batch_dir = os.path.join(batch_text_dir, f"batch_{batch_number}")
    new_file_flag = True  # Reset file numbering for each batch

    for idx, (turn, _, _) in enumerate(batch_segments):
        start_time = turn.start
        end_time = turn.end
        segment_time = end_time - start_time

        if segment_time < 1:
            continue
        
        print("Processing segment:", idx + 1)
        # Extract audio segment
        audio_segment_path = extract_audio_segment(audio_path, start_time, end_time, batch_dir, new_file_flag)

        # Transcribe audio segment
        content = transcribe_audio(audio_segment_path)

        if content:
            print("Fragmented and aligned transcript...")
            fragment_and_align_transcript(content, start_time, batch_dir, new_file_flag)
            new_file_flag = False

        # Remove temporary audio segment
        # os.remove(audio_segment_path)

    print(f"Batch {batch_number} processing completed.")

def detect_speech_segments(file_path,run_folder_dir):
    # Convert the audio file to 16000 Hz and mono
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    
    # Save the converted audio to a temporary file
    temp_file_path = os.path.join(run_folder_dir, "temp_converted_audio.wav")
    audio.export(temp_file_path, format="wav")

    # Use Demucs to separate vocals from background music
    #subprocess.run(["demucs", "-o", run_folder_dir, temp_file_path], check=True)

    #temp_file_path = os.path.join(run_folder_dir, "htdemucs", "temp_converted_audio", "vocals.wav")
    print('Starting diarization')
    print(temp_file_path)

    # Process the converted audio file with the pipeline to get diarization results
    diarization = pipeline(temp_file_path)
    print('Diarization completed')    
    
    # Collect all segments from diarization
    segments = [(turn, _, _) for turn, _, _ in diarization.itertracks(yield_label=True)]

    # Number of threads/batches
    num_processes = 20

    # Calculate number of segments per batch
    num_segments = len(segments)
    segments_per_batch = num_segments // num_processes
    remainder = num_segments % num_processes

    # Create batch folders
    batch_text_dir = os.path.join(run_folder_dir, "batch_text")
    os.makedirs(batch_text_dir, exist_ok=True)

    batches = []
    start_idx = 0
    for i in range(num_processes):
        # Calculate batch size
        if i < remainder:
            batch_size = segments_per_batch + 1
        else:
            batch_size = segments_per_batch

        batch_segments = segments[start_idx:start_idx + batch_size]
        batches.append(batch_segments)  # Include batch number
        start_idx += batch_size

        # Create batch folder
        batch_dir = os.path.join(batch_text_dir, f"batch_{i+1}")
        os.makedirs(batch_dir, exist_ok=True)

    # Process batches concurrently using multiprocessing
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(process_batch, [(batches[i], i + 1) for i in range(num_processes)])

    # After all batches are processed, renumber files and copy to final_segmented folder
    final_segmented_dir = os.path.join(run_folder_dir, "final_segmented")
    os.makedirs(final_segmented_dir, exist_ok=True)

    file_counter = 1
    for i in range(num_processes):
        batch_dir = os.path.join(batch_text_dir, f"batch_{i+1}")
        text_files = sorted([f for f in os.listdir(batch_dir) if f.endswith('.txt')])

        for text_file in text_files:
            # Copy and rename text file
            src_text_path = os.path.join(batch_dir, text_file)
            dst_text_path = os.path.join(final_segmented_dir, f"audio_{file_counter}.txt")
            shutil.copy(src_text_path, dst_text_path)

            # Copy and rename corresponding audio file
            audio_file = text_file.replace('.txt', '.wav')
            src_audio_path = os.path.join(batch_dir, audio_file)
            dst_audio_path = os.path.join(final_segmented_dir, f"audio_{file_counter}.wav")
            shutil.copy(src_audio_path, dst_audio_path)

            file_counter += 1

    print("Created all segmentation!")
    # Get the directory of the current script
    current_dir = os.path.dirname(__file__)

    # Modify the paths to be relative to the current directory
    dictionary_path = os.path.join(current_dir, "dictionary.dict")
    acoustic_model_path = os.path.join(current_dir, "model.zip")

    final_segmented_dir = os.path.join(run_folder_dir, "final_segmented")
    os.makedirs(final_segmented_dir, exist_ok=True)

    # Get the number of CPU cores
    num_cores = os.cpu_count()

    # Construct the MFA command
    mfa_command = [
        "mfa", "align",
        final_segmented_dir,  # Corpus directory
        dictionary_path,  # Dictionary path
        acoustic_model_path,  # Acoustic model path
        final_segmented_dir ,  # Output directory, can be the same or different
        '--num_jobs', str(num_cores),  # Number of jobs to run in parallel        
        '--single_speaker',  # Single speaker mode
        '--verbose',
        '--beam', '200',  # Beam size
        '--retry_beam', '220',  # Retry beam size
    ]

    #startupinfo = subprocess.STARTUPINFO()
    #startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    #startupinfo.wShowWindow = subprocess.SW_HIDE

    subprocess.run(mfa_command, check=True)

    
    # Convert the .textgrid files to .srt files and save them in the sub_segmented folder
    sub_segmented_dir = os.path.join(run_folder_dir, "sub_segmented")
    os.makedirs(sub_segmented_dir, exist_ok=True)


    # Read the initial times from mfa_time_log.txt
    time_log_path = os.path.join(run_folder_dir, "mfa_time_log.txt")
    with open(time_log_path, "r") as log_file:
        initial_times = log_file.readlines()

    # Convert times to floats and sort them in ascending order
    initial_times = [float(time.strip()) for time in initial_times]
    initial_times.sort()
    

    # Traverse through the files in final_segmented directory
    srt_number=1
    for i in range(1, len(initial_times) + 1):
        textgrid_file_name = f"audio_{i}.TextGrid"
        textgrid_file_path = os.path.join(final_segmented_dir, textgrid_file_name)
    
        # Check if the textgrid file exists
        if not os.path.exists(textgrid_file_path):
            print(f"TextGrid file not found: {textgrid_file_path}")
            continue
    
        srt_file_path = os.path.join(sub_segmented_dir, f"audio_{srt_number}.srt")
        srt_number+=1

        # Get the initial_time from the corresponding line in mfa_time_log.txt
        initial_time = float(initial_times[i - 1].strip())
    
        # Assuming convert_textgrid_to_srt is a function you have that does the conversion
        convert_textgrid_to_srt(textgrid_file_path, srt_file_path)
    
        # Add start_time to each timestamp in the srt file
        add_start_time_to_srt(srt_file_path, initial_time)
    
    print("Converted all TextGrid files to SRT files!")
    
    

def generate_srt(audio_file_path, output_srt_path):
    # Get the directory where the script is currently running
    current_working_dir = os.path.dirname(os.path.abspath(__file__))
    run_folder_dir = os.path.join(current_working_dir, "run")
    
    #run_folder_dir='D:\\run'

    if os.path.exists(run_folder_dir):
        shutil.rmtree(run_folder_dir)

    os.makedirs(run_folder_dir, exist_ok=True)

    print("Speaker diarization in progress...")
    detect_speech_segments(audio_file_path,run_folder_dir)
    
   
    # Directory setup for sub_segmented
    sub_segmented_path = os.path.join(run_folder_dir, "sub_segmented")

    # Create sub_segmented directory if it doesn't exist
    if not os.path.exists(sub_segmented_path):
        os.makedirs(sub_segmented_path)
        print(f"Created directory: {sub_segmented_path}")

    # Combine all SRT files in sub_segmented folder into temp_srt_path
    temp_srt_path = os.path.join(run_folder_dir, "temp.srt")
    counter = 1

    with open(temp_srt_path, 'w', encoding='utf-8', newline='\r\n') as final_srt:
    # List all .srt files and count them
        srt_files = [f for f in os.listdir(sub_segmented_path) if f.endswith('.srt')]
        num_files = len(srt_files)

        for i in range(1, num_files + 1):
            srt_file = f"audio_{i}.srt"
            srt_file_path = os.path.join(sub_segmented_path, srt_file)
            if os.path.exists(srt_file_path):
                with open(srt_file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        if line.strip().isdigit():
                            line = f"{counter}\n"
                            counter += 1
                        final_srt.write(line)
                    final_srt.write('\n')
                    final_srt.flush()
    
    # Copy content from temp_srt_path to output_srt_path
    shutil.copy(temp_srt_path, output_srt_path)
    print(f"Copied content from {temp_srt_path} to {output_srt_path}")

    # Post-process the SRT file
    postprocess_srt(output_srt_path)

    #Check for overlapping entries in the SRT file
    check_overlapping_entries(output_srt_path)

    print("SRT file generated successfully!")
    #shutil.rmtree("run")

# Example usage
if __name__ == "__main__":
    # Run main.py
    subprocess.run(["python3", "main.py"])
