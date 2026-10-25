# 目录整理前的冻结快照

2026-10-25，用户要求把本次实验改为 data_recipe、training_recipe、evidence 和 results 四个子目录。这里保留迁移前的文件，用于追溯本次外层冻结身份变化，不作为现行启动入口。

原训练冻结清单 SHA256 为 `debe578ea9a6280297b71e84fb32438109e10acddbfe18514fdfd809736d0e56`，其核心运行合同为 `b8c2d23cbecfb1d0c92aaa00c40ffb0d0af9ef0a379bae1b487ac9de23d4bf58`。

本目录的 training_freeze.json、TRAINING_CONTRACT.md、training_config.json、contract.json、preflight.json 与 original_README.md 是旧版原样快照。里面的相对路径按原位置解释；保留其历史字节，不为当前目录重写。file_inventory.json 记录迁移前的小型数据及短跑产物哈希和新旧位置，文档导航调整会单独改变 README 内容。

当前采用版本见[实验入口](../../README.md)和[现行训练配方](../../training_recipe/TRAINING_CONTRACT.md)。整理没有改变真实 JSONL、tokenizer、核心训练代码、配置参数或 E 盘 checkpoint。现行外层冻结清单会重新记录文档及引用路径，其核心运行合同仍须与原短跑一致。

迁移后已核验：31 份机器证据字节不变，最终 training_config.json 与 contract.json 与旧快照相同；现行归档的 34 个本地文档链接全部有效。新外层冻结清单 SHA256 为 `b496d4fb257bae495d6a3a86552d948206227bd5a8a78a08fbccfd425cc35828`，13 个引用及实际数据、代码、环境和初始化门禁通过，结果见 [results/preflight.json](../../results/preflight.json)。
