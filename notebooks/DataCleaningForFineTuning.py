import json
from pathlib import Path

# (Keep your read_whatsapp_chat function as-is above)

# Read all WhatsApp chat files into dictionary
all_chats = {}
data_directory = Path("../data/private")
for file in data_directory.glob("*.txt"):
    file_name = file.stem
    all_chats[file_name] = read_whatsapp_chat(file)

#  Combine messages with sender info
text_sequence = ""
structured_data = []  # for saving to JSON

for file_name, chat_df in all_chats.items():
    text_sequence += f"\n--- Start of chat: {file_name} ---\n"
    structured_chat = []

    for _, row in chat_df.iterrows():
        sender = row["sender"].strip() if pd.notna(row["sender"]) else "Unknown"
        message = row["message"].strip() if pd.notna(row["message"]) else ""

        # Add line to text file format
        text_sequence += f"{sender}: {message}\n"

        # Add structured JSON entry
        structured_chat.append({
            "sender": sender,
            "message": message
        })

    text_sequence += f"--- End of chat: {file_name} ---\n\n"
    structured_data.append({
        "chat_name": file_name,
        "messages": structured_chat
    })

# Save combined text file
output_text_path = Path("../output/combined_text.txt")
with open(output_text_path, "w", encoding="utf-8") as f:
    f.write(text_sequence)

# Save structured JSON file
output_json_path = Path("../output/combined_text.json")
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(structured_data, f, ensure_ascii=False, indent=4)

print(f"Saved structured chat data to {output_json_path}")
print(f"Saved combined text data to {output_text_path}")
