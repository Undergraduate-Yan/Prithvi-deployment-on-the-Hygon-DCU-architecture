# K100-2 同 CLI：M5 与 FP16 full 配对性能协议

本目录只新增一套独立证据，不读取或覆盖旧性能结果。正式计时对象是两个已冻结部署 bundle：

- M5 manifest SHA256：`f4dc433595c43332d4e29c2351363b3245f214273f058551788d50be29c388da`
- FP16 full manifest SHA256：`1f433a917aed5fead6dd95597a5a03f500a93ce564067e9dad4d25f4682bad25`
- 官方镜像 ID：`sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01`

## 协议

每个候选在 3 个全新 Docker 容器进程中分别执行，因此总计 6 个计时容器。每个容器先完成 bundle 全文件 SHA256 验证和 MXR/session 加载，再执行 30 次预热与 100 次正式测量。三轮启动顺序为：

1. M5 → FP16 full；
2. FP16 full → M5；
3. M5 → FP16 full。

由于轮数为奇数，这是位置次数差不超过 1 的最大平衡设计。每个容器只常驻一个模型，避免 M5 与 FP16 同时常驻导致额外显存竞争。

两者都从各自 bundle 中导入同一个 SHA 锁定的 `infer_k100.py`，并调用 `K100BundleRunner.run`。正式时间直接采用其第三个返回值：float32 `1×6×224×224` 固定输入到已经同步并回到 host 的 float32 logits。完整性验证、session/cache 加载、argmax、prediction SHA 和证据写盘均不计时。

工具逐次拒绝以下漂移：错误镜像、错误 manifest、bundle 任意 payload SHA 变化、错误 Docker-level PATH、static lsmod 变化、非 MIGraphX 首选 provider、CPU fallback 配置不是 `1`、输入变化、130 次预测中的任意一次变化，以及输出目录已存在。两个 bundle 均以只读方式挂载。公平性要求两个 bundle 的 raw input array SHA、shape 和 dtype 完全一致；M5 与 FP16 各自保留自己的冻结 prediction oracle，二者的 expected prediction SHA 不要求相同，但每次输出都必须匹配所属 bundle 的 oracle。

## 上传后在 K100-2 执行

将本目录原样上传为 `/var/tmp/phase11_deployment_cli_pair_benchmark_20260818` 后，确认 K100-2 没有其他容器或 GPU 任务，再运行一条命令：

```bash
chmod 0555 /var/tmp/phase11_deployment_cli_pair_benchmark_20260818/*.py /var/tmp/phase11_deployment_cli_pair_benchmark_20260818/*.sh && /var/tmp/phase11_deployment_cli_pair_benchmark_20260818/run_k100_2_bundle_cli_pair.sh /var/tmp/20260818-phase11-minimal-deployment-node2-v1/bundles/prithvi-k100-m5-v1 /var/tmp/20260818-phase11-minimal-deployment-node2-v1/bundles/prithvi-k100-fp16-full-v1 /var/tmp/20260818-phase11-minimal-deployment-node2-v1/receipts/paired_cli_m5_vs_fp16_v1
```

输出目录必须事先不存在。正式汇总是 `summary.json`，原始 600 个时延、六个 Docker inspect、容器日志和文件 SHA 清单均保留在同一新目录。

如果六个 trial 已经成功、只有旧聚合器因错误要求跨 bundle prediction SHA 相同而失败，则不需要重跑 GPU。上传新版 `aggregate_bundle_cli_pair.py` 后直接执行：

```bash
test ! -e /var/tmp/20260818-phase11-minimal-deployment-node2-v1/receipts/paired_cli_m5_vs_fp16_v1/summary.json && /usr/bin/python3 /var/tmp/phase11_deployment_cli_pair_benchmark_20260818/aggregate_bundle_cli_pair.py --run-root /var/tmp/20260818-phase11-minimal-deployment-node2-v1/receipts/paired_cli_m5_vs_fp16_v1 --expected-trial-script-sha256 da7ae2c77e4b469ba0674751e996a5dfcaf0226a638c90b364e3a3503e821560 --output /var/tmp/20260818-phase11-minimal-deployment-node2-v1/receipts/paired_cli_m5_vs_fp16_v1/summary.json
```

## 结果判定

- M5、FP16 full 的 3 个 trial median CV 都必须 `≤5%`，否则结果仅为诊断，禁止正式速度表述。
- 稳定性门槛通过后，按全部 300 次测量的 median 计算 `M5 / FP16`。
- `M5 / FP16 ≤1.05` 才通过“同 CLI 时延不比 FP16 full 慢超过 5%”升级条件。
- 此工具不证明 INT8 kernel 精度，不覆盖历史 strict-logits 失败，也不会单独改变部署推荐。

## 工具身份

| 文件 | 字节 | SHA256 |
|---|---:|---|
| `benchmark_bundle_cli_trial.py` | 18146 | `da7ae2c77e4b469ba0674751e996a5dfcaf0226a638c90b364e3a3503e821560` |
| `aggregate_bundle_cli_pair.py` | 26628 | `5904f9c7f2f4a93c0613746a5c780021b02a486155f068921bf42e23c8dbaf5c` |
| `run_k100_2_bundle_cli_pair.sh` | 7185 | `408eb0d63a7d327fd5633160dd8dd9d585f073683bff853d407119dde0e49e2b` |

CPU-only 回归：

```bash
python3 -B test_bundle_cli_pair_static.py
```
