# superpowers（本项目局部安装）

本目录包含 [obra/superpowers](https://github.com/obra/superpowers) **v6.4.2**（2026-09-25 发布，MIT 许可证）的工作区级安装，**只对 Mewhelp 这个项目生效**。

## 为什么不通过插件市场安装

ZCode 的插件管理（市场安装 + 启用状态）只有用户级作用域，装了会影响所有项目；且 `claude-plugins-official` 市场里固定的 superpowers 版本是 v6.3.0。要"只装本项目 + 指定版本"，按 ZCode 的工作区发现规则安装组件是官方支持的路径（工作区 skills / hooks）。

## 安装内容

| 文件/目录 | 来源 | 作用 |
|---|---|---|
| `skills/`（15 个技能目录） | 上游 `skills/` | brainstorming、writing-plans、executing-plans、subagent-driven-development、test-driven-development、systematic-debugging 等 |
| `hooks/session-start` | 上游 `hooks/session-start` | SessionStart hook，把 `using-superpowers` 技能内容注入会话上下文（脚本按自身位置推导根目录，读取 `../skills/using-superpowers/SKILL.md`，因此放在 `.zcode/hooks/` 下即可工作） |
| `config.json` | 本项目手写 | 启用工作区 hooks：SessionStart（matcher `startup\|clear\|compact`）时以 `bash` 运行上述脚本，超时 10 秒 |

上游的 `hooks/run-hook.cmd`（cmd/bash 多语言包装器）未采用——它服务于无 Git Bash 的 Windows 环境；本机有 Git Bash，直接 `bash` 调用更简单可靠。

## 与上游的差异

- skills 正文和 hook 注入文本中的 `superpowers:` 技能名前缀已被移除（如 `superpowers:brainstorming` → `brainstorming`），因为工作区技能注册名不带命名空间，改写后技能间引用可精确解析。

## 如何更新版本

1. 下载新版本源码：`https://codeload.github.com/obra/superpowers/tar.gz/refs/tags/vX.Y.Z` 并解压。
2. 覆盖复制：`skills/.` → 本目录 `skills/`；`hooks/session-start` → 本目录 `hooks/session-start`。
3. 重新应用前缀改写：`sed -i 's/superpowers://g' .zcode/skills/*/SKILL.md .zcode/hooks/session-start`（在项目根目录执行）。

## 如何卸载

删除 `.zcode/skills/`、`.zcode/hooks/`，并从 `.zcode/config.json` 移除整个 `hooks` 块即可。

## 验证方式

重新打开本项目的会话后：`using-superpowers` 等 15 个技能应出现在技能列表（Settings → Skills 或 `/` 菜单）；SessionStart 时日志中应有该 hook 的成功执行记录。
