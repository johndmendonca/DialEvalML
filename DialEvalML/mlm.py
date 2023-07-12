import torch


class MLM_predictor:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, seg):
        stride = 8
        with torch.no_grad():
            inputs = self.tokenizer(seg, truncation=True, return_tensors="pt")

            input_ids = inputs["input_ids"].clone()
            input_ids[0][1] = self.tokenizer.mask_token_id

            labels = torch.where(
                input_ids == self.tokenizer.mask_token_id, inputs["input_ids"], -100
            )

            for i in range(2, len(inputs["input_ids"][0]) - 1):
                local_input = inputs["input_ids"].clone()
                local_input[0][i] = self.tokenizer.mask_token_id

                input_ids = torch.cat((input_ids, local_input))
                labels = torch.cat(
                    (
                        labels,
                        torch.where(
                            local_input == self.tokenizer.mask_token_id,
                            inputs["input_ids"],
                            -100,
                        ),
                    )
                )

            attention_mask = inputs["attention_mask"].repeat(len(input_ids), 1)

            loss = 0
            for i in range(0, len(input_ids), stride):
                loss += self.model(
                    input_ids=input_ids[i : i + stride].to(self.device),
                    attention_mask=attention_mask[i : i + stride].to(self.device),
                    labels=labels[i : i + stride].to(self.device),
                ).loss.item()

            return loss
