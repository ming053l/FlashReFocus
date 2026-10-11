"""CSD teacher prompts for Stage I training (Sec. 3.1): RAM++ tags of the sharp target, as
"A sharp clear image, containing <tag>, <tag>, ...", or a fallback prompt when no tag is active."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from flashrefocus import paths

PROMPT_PREFIX = "A sharp clear image, containing "


class RamPlusPromptTagger(nn.Module):
    def __init__(self, checkpoint=paths.RAM_PLUS, image_size=384, threshold=0.7, max_tags=50,
                 fallback_prompt="a sharp, clear image"):
        super().__init__()
        paths.use_third_party(paths.PISA_SR, paths.RAM_PLUS_CODE)    # RAM++ must shadow PiSA-SR's older `ram`
        from ram.models import ram_plus
        self.model = ram_plus(pretrained=str(checkpoint), image_size=image_size, vit="swin_l")
        self.image_size, self.threshold, self.max_tags, self.fallback_prompt = image_size, threshold, max_tags, fallback_prompt
        nn.Linear(512, 1024)   # unused: draws the random numbers of a layer of the released training, which keeps its stream
        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def _tag_probabilities(self, images):
        """images (b, 3, h, w) in [0, 1] -> probability of every RAM++ tag (b, num_tags)."""
        x = F.interpolate(images.float().clamp(0.0, 1.0), size=(self.image_size, self.image_size), mode="bilinear",
                          align_corners=False)
        x = (x.to(self.pixel_mean) - self.pixel_mean) / self.pixel_std
        m = self.model
        image_embeds = m.image_proj(m.visual_encoder(x))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        cls = image_embeds[:, 0, :]
        cls = cls / cls.norm(dim=-1, keepdim=True)
        per_class = m.label_embed.shape[0] // m.num_class
        logits = (m.reweight_scale.exp() * cls @ m.label_embed.t()).view(images.shape[0], -1, per_class)
        label_embed = torch.einsum("bcd,cdh->bch", F.softmax(logits, dim=2), m.label_embed.view(-1, per_class, 512))
        tagging = m.tagging_head(encoder_embeds=torch.relu(m.wordvec_proj(label_embed)),
                                 encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts,
                                 return_dict=False, mode="tagging")
        return torch.sigmoid(m.fc(tagging[0]).squeeze(-1))

    @torch.no_grad()
    def prompts(self, images):
        """One prompt per image of a batch in [0, 1]."""
        self.eval()
        probabilities = self._tag_probabilities(images).cpu()
        tags = np.asarray(self.model.tag_list)
        prompts = []
        for p in probabilities:
            active = p > self.threshold
            active[list(self.model.delete_tag_index)] = False
            indices = torch.where(active)[0].tolist()[:self.max_tags]
            prompts.append(PROMPT_PREFIX + ", ".join(tags[indices].tolist()) if indices else self.fallback_prompt)
        return prompts
