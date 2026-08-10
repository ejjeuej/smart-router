# Hermes 源码改动记录

git merge / pull 上游后，按此文件重新打 patch。

---

## 1. 状态栏同步模型名

**文件：** `agent/conversation_loop.py`  
**行：** 1519-1523  
**用途：** middleware 切换模型后，把真实模型名写回 agent.model，CLI/TUI 状态栏显示切换后的模型。

**git 补丁：**
```diff
@@ -1516,6 +1516,11 @@
                     thinking_spinner = None
                 if agent.thinking_callback:
                     agent.thinking_callback("")
+                # Sync agent.model to middleware-routed model so the
+                # status bar shows the model that actually answered.
+                _resp_model = getattr(response, "model", None) if response else None
+                if _resp_model and _resp_model != agent.model:
+                    agent.model = _resp_model

                 if not agent.quiet_mode:
```

**手动恢复方法：** 在 `conversation_loop.py` 的 `agent.thinking_callback("")` 之后、`if not agent.quiet_mode:` 之前插入以上 5 行。

---

## 2. 允许 Agent 写 config.yaml

**文件：** `tools/file_tools.py`（`_check_sensitive_path` + 新增 `_allow_agent_config_write`）  
**文件：** `hermes_cli/config.py`（新增 `security.allow_agent_config_write` 配置项）  
**用途：** Hermes 默认禁止 agent 修改 config.yaml（安全保护）。加了 `security.allow_agent_config_write: true` 开关，在 config.yaml 打开后 agent 就能写。

**手动恢复方法：**
- `file_tools.py`：把 `_check_sensitive_path` 里的硬拒绝改为用 `_allow_agent_config_write()` 判断，并新增该函数
- `config.py`：在 `DEFAULT_CONFIG["security"]` 下添加 `"allow_agent_config_write": False` 字段

**用户 config 开关：** `~/.hermes/config.yaml` → `security.allow_agent_config_write: true`


---

## 3. smart-router 插件配置

**文件：** `~/.hermes/config.yaml`  
**用途：** 路由插件的配置段，不在源码仓库里，但升级 Hermes 不影响 config.yaml。

```yaml
plugins:
  enabled:
  - smart-router

smart_model_routing:
  enabled: true
  simple_models:
  - deepseek-chat
  complex_models:
  - deepseek-v4-pro
  providers:
    deepseek-chat:
      base_url: https://api.deepseek.com/v1
      api_key_env: DEEPSEEK_API_KEY
    deepseek-v4-pro:
      base_url: https://api.deepseek.com/v1
      api_key_env: DEEPSEEK_API_KEY
```

---

## 恢复流程（git merge 后）

```bash
# 1. 确认哪些文件被上游覆盖了
cd ~/Hermes_code/hermes-agent
git diff main...HEAD -- agent/conversation_loop.py tools/file_tools.py hermes_cli/config.py toolsets.py hermes_cli/tools_config.py

# 2. 按上面各条逐一恢复
```
