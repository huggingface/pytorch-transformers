<!---
Copyright 2020 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# 範例

我們提供了針對多種學習框架的範例腳本。請選擇您喜愛的框架：[TensorFlow](https://github.com/huggingface/transformers/tree/main/examples/tensorflow)、[PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch) 或 [JAX/Flax](https://github.com/huggingface/transformers/tree/main/examples/flax)。

我們也有一些[研究專案](https://github.com/huggingface/transformers/tree/main/examples/research_projects)，以及一些[舊版範例](https://github.com/huggingface/transformers/tree/main/examples/legacy)。需要注意的是，這些舊版範例並未被積極維護，可能需要特定舊版的相依套件才能執行。

雖然我們致力於涵蓋盡可能多的使用案例，這些範例腳本僅僅是範例。它們預期不會直接適用於您的特定問題，您可能需要修改部分程式碼以適應您的需求。為了幫助您實現這一點，大多數範例完全公開了資料的預處理過程，您可以根據需求進行調整和編輯。

在提交 PR 之前，請在[論壇](https://discuss.huggingface.co/)或[問題頁面](https://github.com/huggingface/transformers/issues)上討論您希望在範例中實現的功能；我們歡迎錯誤修正，但由於我們希望範例保持簡單，因此不太可能合併以可讀性為代價增加更多功能的 PR。

## 重要注意事項

**重要**

為確保您可以成功執行最新版本的範例腳本，您需要**從原始碼安裝程式庫**並安裝一些特定範例所需的相依套件。請在新的虛擬環境中執行以下步驟：
```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```
然後進入您選擇的範例資料夾，並執行
```bash
pip install -r requirements.txt
```

若要瀏覽與已發布版本的 🤗 Transformers 對應的範例，請點擊下方的連結，然後選擇您想要的程式庫版本：

<details>
  <summary>舊版 🤗 Transformers 的範例</summary>
	<ul>
	    <li><a href="https://github.com/huggingface/transformers/tree/v4.21.0/examples">v4.21.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.20.1/examples">v4.20.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.19.4/examples">v4.19.4</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.18.0/examples">v4.18.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.17.0/examples">v4.17.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.16.2/examples">v4.16.2</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.15.0/examples">v4.15.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.14.1/examples">v4.14.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.13.0/examples">v4.13.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.12.5/examples">v4.12.5</a></li>
	    <li><a href="https://github.com/huggingface/transformers/tree/v4.11.3/examples">v4.11.3</a></li>
	    <li><a href="https://github.com/huggingface/transformers/tree/v4.10.3/examples">v4.10.3</a></li>
	    <li><a href="https://github.com/huggingface/transformers/tree/v4.9.2/examples">v4.9.2</a></li>
	    <li><a href="https://github.com/huggingface/transformers/tree/v4.8.2/examples">v4.8.2</a></li>
	    <li><a href="https://github.com/huggingface/transformers/tree/v4.7.0/examples">v4.7.0</a></li>
	    <li><a href="https://github.com/huggingface/transformers/tree/v4.6.1/examples">v4.6.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.5.1/examples">v4.5.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.4.2/examples">v4.4.2</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.3.3/examples">v4.3.3</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.2.2/examples">v4.2.2</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.1.1/examples">v4.1.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.0.1/examples">v4.0.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.5.1/examples">v3.5.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.4.0/examples">v3.4.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.3.1/examples">v3.3.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.2.0/examples">v3.2.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.1.0/examples">v3.1.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.0.2/examples">v3.0.2</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.11.0/examples">v2.11.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.10.0/examples">v2.10.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.9.1/examples">v2.9.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.8.0/examples">v2.8.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.7.0/examples">v2.7.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.6.0/examples">v2.6.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.5.1/examples">v2.5.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.4.0/examples">v2.4.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.3.0/examples">v2.3.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.2.0/examples">v2.2.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.1.0/examples">v2.1.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.0.0/examples">v2.0.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v1.2.0/examples">v1.2.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v1.1.0/examples">v1.1.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v1.0.0/examples">v1.0.0</a></li>
	</ul>
</details>

或者，您可以將已複製的 🤗 Transformers 切換到特定版本（例如 v3.5.1）：
```bash
git checkout tags/v3.5.1
```
然後按照通常的方式執行範例命令。

## 使用自動設定在遠端硬體上執行範例

[run_on_remote.py](./run_on_remote.py) 是一個用於在遠端自託管硬體上啟動任意範例的腳本，具有自動硬體與環境設定功能。它使用 [Runhouse](https://github.com/run-house/runhouse) 在自託管硬體（例如，您自己的雲帳戶或內部集群）上啟動，但也有其他選項可以遠端執行。

您可以輕鬆自訂使用的範例、命令行參數、相依套件和計算硬體類型，然後執行腳本自動啟動範例。

您可以參考 [硬體設定](https://www.run.house/docs/tutorials/quick-start-cloud) 以了解 Runhouse 的硬體與相依套件設定，或參考這個 [Colab 教程](https://colab.research.google.com/drive/1sh_aNQzJX5BKAdNeXthTNGxKz7sM9VPc) 以獲得更深入的說明。

您可以使用以下命令執行腳本：

```bash
# 首先安裝 runhouse：
pip install runhouse

# 使用您已配置的雲供應商的隨選 V100：
python run_on_remote.py \
    --example pytorch/text-generation/run_generation.py \
    --model_type=gpt2 \
    --model_name_or_path=openai-community/gpt2 \
    --prompt "I am a language model and"

# 用於自帶（bring your own）集群：
python run_on_remote.py --host <cluster_ip> --user <ssh_user> --key_path <ssh_key_path> \
  --example <example> <args>

# 用於隨選實例：
python run_on_remote.py --instance <instance> --provider <provider> \
  --example <example> <args>
```

您也可以根據自己的需求自訂腳本。
