---
license: mit
dataset_info:
  features:
  - name: question_id
    dtype: string
  - name: cluster_name
    dtype: string
  - name: turns
    list:
      list:
      - name: content
        dtype: string
  - name: images
    sequence: image
  splits:
  - name: train
    num_bytes: 279845451.0
    num_examples: 500
  download_size: 277821717
  dataset_size: 279845451.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
task_categories:
- visual-question-answering
size_categories:
- 100<n<1K
---

![Vision Arena Questions](vision_arena_questions_fig.png)

# VisionArena-Bench: An automatic eval pipeline to estimate model preference rankings

An automatic benchmark of 500 diverse user prompts that can be used to cheaply approximate [Chatbot Arena](https://lmarena.ai/) model rankings via automatic benchmarking with VLM as a judge. 

### Dataset Sources

- **Repository:** https://github.com/lm-sys/FastChat
- **Paper:** https://arxiv.org/abs/2412.08687
- **Automatic Evaluation Code:** Coming Soon!

## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

- question_id: The unique hash representing the id of the question
- cluster_name: The name of the topic cluster that this question is from
- turns: The content with the question prompt
- images: A list of images of size one (single-image) which correspond to the question in the column `turns`

## Bias, Risks, and Limitations

This benchmark is designed to measure human preferences rather than explicitly evaluate factual accuracy.

This dataset contains a large amount of STEM related questions, OCR tasks, and general problems like captioning. This dataset contains less questions which relate to specialized domains outside of stem. 

**If you find your face or personal information in this dataset and wish to have it removed, or if you find hateful or inappropriate content,** please contact us at lmarena.ai@gmail.com or lisabdunlap@berkeley.edu.

**BibTeX:**

```
@misc{chou2024visionarena,
      title={VisionArena: 230K Real World User-VLM Conversations with Preference Labels}, 
      author={Christopher Chou and Lisa Dunlap and Koki Mashita and Krishna Mandal and Trevor Darrell and Ion Stoica and Joseph E. Gonzalez and Wei-Lin Chiang},
      year={2024},
      eprint={2412.08687},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.08687}, 
}
```

## LMArena VisionArena dataset License Agreement
This Agreement contains the terms and conditions that govern your access and use of the LMArena VisionArena dataset (as defined above). You may not use the LMArena VisionArena dataset if you do not accept this Agreement. By clicking to accept, accessing the LMArena VisionArena dataset, or both, you hereby agree to the terms of the Agreement. If you are agreeing to be bound by the Agreement on behalf of your employer or another entity, you represent and warrant that you have full legal authority to bind your employer or such entity to this Agreement. If you do not have the requisite authority, you may not accept the Agreement or access the LMArena VisionArena dataset on behalf of your employer or another entity.

* Safety and Moderation: This dataset contains unsafe conversations that may be perceived as offensive or unsettling. User should apply appropriate filters and safety measures before utilizing this dataset for training dialogue agents.
* Non-Endorsement: The views and opinions depicted in this dataset do not reflect the perspectives of the researchers or affiliated institutions engaged in the data collection process.
* Legal Compliance: You are mandated to use it in adherence with all pertinent laws and regulations.
* Model Specific Terms: When leveraging direct outputs of a specific model, users must adhere to its corresponding terms of use.
* Non-Identification: You must not attempt to identify the identities of individuals or infer any sensitive personal data encompassed in this dataset.
* Prohibited Transfers: You should not distribute, copy, disclose, assign, sublicense, embed, host, or otherwise transfer the dataset to any third party.
* Right to Request Deletion: At any time, we may require you to delete all copies of the conversation dataset (in whole or in part) in your possession and control. You will promptly comply with any and all such requests. Upon our request, you shall provide us with written confirmation of your compliance with such requirement.
* Termination: We may, at any time, for any reason or for no reason, terminate this Agreement, effective immediately upon notice to you. Upon termination, the license granted to you hereunder will immediately terminate, and you will immediately stop using the LMArena VisionArena dataset and destroy all copies of the LMArena VisionArena dataset and related materials in your possession or control.
* Limitation of Liability: IN NO EVENT WILL WE BE LIABLE FOR ANY CONSEQUENTIAL, INCIDENTAL, EXEMPLARY, PUNITIVE, SPECIAL, OR INDIRECT DAMAGES (INCLUDING DAMAGES FOR LOSS OF PROFITS, BUSINESS INTERRUPTION, OR LOSS OF INFORMATION) ARISING OUT OF OR RELATING TO THIS AGREEMENT OR ITS SUBJECT MATTER, EVEN IF WE HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
* Subject to your compliance with the terms and conditions of this Agreement, we grant to you, a limited, non-exclusive, non-transferable, non-sublicensable license to use the LMArena VisionArena dataset, including the conversation data and annotations, to research, develop, and improve software, algorithms, machine learning models, techniques, and technologies for both research and commercial purposes.