import json

with open("dataset/validated/golden_dataset_final_cleaned.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

false_roles = 0
false_pair = 0
append = True
cleaned_data = []
for sample in dataset:
    messages = sample["messages"]

    for i, message in enumerate(messages):
        role = message["role"]

        # your logic here
        if role == "assistant":
            content = message.get("content", "")
            if "<tool_call>" in content:
                try:
                    next_message = messages[i + 1]

                    if next_message["role"] == "user":
                        false_roles += 1
                        next_message["role"] = "tool"
                except IndexError as ie:
                    false_pair +=1
                    append = False
    if append:
        cleaned_data.append({"messages": messages})
    append = True
                    

print(false_roles)
print(false_pair)

with open("dataset/validated/golden_dataset_final_cleaned_v2.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(cleaned_data, indent=4))