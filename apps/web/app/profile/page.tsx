"use client";

import { useMutation, useQueries, useQueryClient } from "@tanstack/react-query";
import { Download, FileUp, Plus, Save, Trash2 } from "lucide-react";
import { ChangeEvent, useEffect, useState } from "react";
import { PageHeader } from "@/components/page-header";
import { ProcessGuide } from "@/components/process-guide";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { api, Experience, ParsedDocument, Profile } from "@/lib/api";

const emptyProfile: Omit<Profile, "id" | "owner_id" | "updated_at"> = {
  full_name: "", current_school: "", current_major: "", degree: "Bachelor",
  gpa: null, gpa_scale: 4, language_scores: {}, target_countries: [], target_fields: [],
  intake: "", budget: null, preferences: {}, confirmed: false, experiences: [],
};

const emptyExperience = (): Experience => ({ kind: "project", title: "", organization: "", start_date: "", end_date: "", description: "", tags: [], confirmed: true });

export default function ProfilePage() {
  const client = useQueryClient();
  const [query, memories] = useQueries({ queries: [
    { queryKey: ["profile"], queryFn: api.profile },
    { queryKey: ["memories"], queryFn: api.memories },
  ] });
  const [form, setForm] = useState(emptyProfile);
  const [notice, setNotice] = useState("");
  const [candidateData, setCandidateData] = useState<Record<string, unknown> | null>(null);
  const [parsedDocument, setParsedDocument] = useState<ParsedDocument | null>(null);
  useEffect(() => {
    if (query.data) {
      setForm({
        full_name: query.data.full_name,
        current_school: query.data.current_school,
        current_major: query.data.current_major,
        degree: query.data.degree,
        gpa: query.data.gpa,
        gpa_scale: query.data.gpa_scale,
        language_scores: query.data.language_scores,
        target_countries: query.data.target_countries,
        target_fields: query.data.target_fields,
        intake: query.data.intake,
        budget: query.data.budget,
        preferences: query.data.preferences,
        confirmed: query.data.confirmed,
        experiences: query.data.experiences,
      });
    }
  }, [query.data]);
  const save = useMutation({
    mutationFn: api.saveProfile,
    onSuccess: (data) => { client.setQueryData(["profile"], data); setNotice("画像已确认并写入可追溯长期记忆。 "); },
    onError: (error) => setNotice(error instanceof Error ? error.message : "保存失败"),
  });
  const upload = useMutation({
    mutationFn: ({ file, kind }: { file: File; kind: string }) => api.uploadDocument(file, kind),
    onSuccess: (data) => { setParsedDocument(data); setCandidateData(data.extracted_data); setNotice(`${data.filename} 已解析，状态：${data.parse_status}。候选信息需要你确认后才能写入画像。`); },
    onError: (error) => setNotice(error instanceof Error ? error.message : "上传失败"),
  });
  const confirmCandidates = useMutation({
    mutationFn: () => parsedDocument ? api.confirmDocument(parsedDocument.id, ["gpa", "gpa_scale", "language_scores"]) : Promise.reject(new Error("没有待确认材料")),
    onSuccess: (data) => { client.setQueryData(["profile"], data); setParsedDocument(null); setCandidateData(null); setNotice("已把材料中明确选择的候选字段写入画像，并保存材料来源；其他经历仍需手工核对。 "); client.invalidateQueries({ queryKey: ["memories"] }); },
    onError: (error) => setNotice(error instanceof Error ? error.message : "候选字段确认失败"),
  });
  const clear = useMutation({
    mutationFn: api.deleteProfile,
    onSuccess: () => {
      setForm(emptyProfile);
      client.invalidateQueries();
      setNotice("单用户空间中的画像、经历、材料版本、任务、选校清单和长期记忆已清空。");
    },
    onError: (error) => setNotice(error instanceof Error ? error.message : "清空失败"),
  });
  const removeMemory = useMutation({
    mutationFn: api.deleteMemory,
    onSuccess: () => { client.invalidateQueries({ queryKey: ["memories"] }); setNotice("这条长期记忆已停用；画像业务数据不会被同步删除。"); },
    onError: (error) => setNotice(error instanceof Error ? error.message : "删除记忆失败"),
  });

  function update<K extends keyof typeof form>(key: K, value: (typeof form)[K]) { setForm((current) => ({ ...current, [key]: value })); }
  function changeExperience(index: number, key: keyof Experience, value: string | boolean | string[]) {
    setForm((current) => ({ ...current, experiences: current.experiences.map((item, itemIndex) => itemIndex === index ? { ...item, [key]: value } : item) }));
  }
  async function onFile(event: ChangeEvent<HTMLInputElement>, kind: string) {
    const file = event.target.files?.[0];
    if (file) upload.mutate({ file, kind });
    event.target.value = "";
  }
  function confirmClear() {
    if (window.confirm("确定清空当前单用户空间中的全部个人申请数据吗？此操作不可撤销。")) clear.mutate();
  }

  return (
    <div className="mx-auto max-w-6xl">
      <PageHeader eyebrow="Applicant memory" title="我的画像" description="这是选校和材料规划的事实底座。只有你确认的信息才会进入长期记忆；AI 抽取结果始终先作为候选展示。" actions={<div className="flex flex-wrap gap-2"><a href={api.profileExportUrl} download><Button variant="secondary"><Download size={16} />导出数据</Button></a><Button variant="secondary" onClick={confirmClear} disabled={clear.isPending}><Trash2 size={16} />清空数据</Button><Button onClick={() => save.mutate({ ...form, confirmed: true })} disabled={save.isPending}><Save size={16} />确认并保存</Button></div>} />
      <ProcessGuide current={0} />
      {notice && <div className="mb-5 rounded-xl border border-moss/15 bg-mint/70 px-4 py-3 text-sm text-moss">{notice}</div>}
      <div className="grid gap-6 xl:grid-cols-[1fr_320px]">
        <div className="space-y-6">
          <Card>
            <p className="eyebrow">Basics</p><h2 className="mt-2 text-xl font-black">学术背景</h2>
            <div className="mt-5 grid gap-4 sm:grid-cols-2">
              {[
                ["姓名", "full_name", "例如：李同学"], ["当前学校", "current_school", "学校全称"],
                ["当前专业", "current_major", "例如：软件工程"], ["当前学位", "degree", "Bachelor / Master"],
              ].map(([label, key, placeholder]) => <label key={key}><span className="label">{label}</span><input className="field" value={String(form[key as keyof typeof form] ?? "")} placeholder={placeholder} onChange={(event) => update(key as "full_name", event.target.value)} /></label>)}
              <label><span className="label">GPA</span><input className="field" type="number" step="0.01" value={form.gpa ?? ""} onChange={(event) => update("gpa", event.target.value ? Number(event.target.value) : null)} /></label>
              <label><span className="label">GPA 满分</span><input className="field" type="number" step="0.1" value={form.gpa_scale ?? ""} onChange={(event) => update("gpa_scale", event.target.value ? Number(event.target.value) : null)} /></label>
              <label><span className="label">TOEFL</span><input className="field" type="number" value={form.language_scores.TOEFL ?? ""} onChange={(event) => update("language_scores", { ...form.language_scores, TOEFL: Number(event.target.value) })} /></label>
              <label><span className="label">IELTS</span><input className="field" type="number" step="0.5" value={form.language_scores.IELTS ?? ""} onChange={(event) => update("language_scores", { ...form.language_scores, IELTS: Number(event.target.value) })} /></label>
            </div>
          </Card>
          <Card>
            <div className="flex items-end justify-between"><div><p className="eyebrow">Evidence library</p><h2 className="mt-2 text-xl font-black">真实经历库</h2></div><Button variant="secondary" onClick={() => update("experiences", [...form.experiences, emptyExperience()])}><Plus size={15} />添加经历</Button></div>
            <div className="mt-5 space-y-4">
              {form.experiences.map((experience, index) => (
                <div key={experience.id ?? index} className="rounded-2xl border border-black/5 bg-paper/60 p-4">
                  <div className="grid gap-3 sm:grid-cols-[130px_1fr_1fr_auto]">
                    <select className="field" value={experience.kind} onChange={(event) => changeExperience(index, "kind", event.target.value)}><option value="project">项目</option><option value="research">科研</option><option value="internship">实习</option><option value="award">奖项</option><option value="course">课程</option></select>
                    <input className="field" placeholder="经历标题" value={experience.title} onChange={(event) => changeExperience(index, "title", event.target.value)} />
                    <input className="field" placeholder="组织 / 实验室 / 公司" value={experience.organization} onChange={(event) => changeExperience(index, "organization", event.target.value)} />
                    <Button variant="ghost" aria-label="删除经历" onClick={() => update("experiences", form.experiences.filter((_, itemIndex) => itemIndex !== index))}><Trash2 size={16} /></Button>
                  </div>
                  <textarea className="field mt-3 min-h-24" placeholder="写清楚你真实做过什么、产生了什么结果。" value={experience.description} onChange={(event) => changeExperience(index, "description", event.target.value)} />
                </div>
              ))}
              {!form.experiences.length && <p className="rounded-xl border border-dashed border-black/10 p-6 text-center text-sm text-ink/45">还没有经历。CV 和 PS Skill 不会在这里为空时编造素材。</p>}
            </div>
          </Card>
        </div>
        <div className="space-y-6">
          <Card><p className="eyebrow">Document parsing</p><h2 className="mt-2 text-xl font-black">上传材料</h2><p className="mt-2 text-sm leading-6 text-ink/50">支持 PDF、DOCX、TXT、Markdown 和图片。抽取信息不会自动覆盖画像。</p>
            <div className="mt-4 space-y-3">
              {[
                ["cv", "上传初始 CV"], ["ps", "上传初始 PS"], ["transcript", "上传成绩单"],
                ["recommendation", "上传推荐信"], ["language", "上传语言成绩"],
                ["writing_sample", "上传 Writing Sample"], ["portfolio", "上传作品集"],
                ["video_essay", "上传 Video Essay 文件"],
              ].map(([kind, label]) => <label key={kind} className="flex cursor-pointer items-center justify-between rounded-xl border border-dashed border-moss/25 bg-mint/40 px-3 py-3 text-sm font-semibold hover:bg-mint"><span>{label}</span><FileUp size={16} /><input type="file" className="hidden" onChange={(event) => onFile(event, kind)} /></label>)}
            </div>
            {candidateData && <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-3"><p className="text-xs font-black text-amber-900">AI / 解析器候选结果（尚未写入画像）</p><p className="mt-2 whitespace-pre-wrap break-words text-xs leading-5 text-amber-800">{JSON.stringify(candidateData, null, 2)}</p><div className="mt-3 flex flex-wrap gap-2"><Button size="sm" variant="secondary" disabled={confirmCandidates.isPending} onClick={() => confirmCandidates.mutate()}>确认 GPA / 语言候选</Button><Button size="sm" variant="ghost" onClick={() => setNotice("经历、教育和奖项请逐项核对后填写到左侧表单；系统不会自动写入。")}>其他字段手工核对</Button></div></div>}
          </Card>
          <Card><p className="eyebrow">Memory policy</p><h2 className="mt-2 text-xl font-black">长期记忆</h2><ul className="mt-4 space-y-3 text-sm leading-6 text-ink/60"><li>• 已确认画像可以长期保存。</li><li>• 文档抽取先作为候选事实。</li><li>• 冲突信息保留版本，不静默覆盖。</li><li>• 你可以导出数据，或清空当前单用户空间。</li></ul><div className="mt-5 space-y-3">{memories.data?.map((memory) => <div key={memory.id} className="rounded-xl border border-black/5 bg-paper/70 p-3"><div className="flex items-start justify-between gap-2"><div><p className="text-xs font-black">{memory.key}</p><p className="mt-1 text-[11px] text-ink/45">{memory.memory_type} · {memory.source_type}</p></div><button aria-label="删除长期记忆" className="text-ink/35 hover:text-red-600" onClick={() => removeMemory.mutate(memory.id)}><Trash2 size={14} /></button></div><p className="mt-2 line-clamp-3 break-all text-xs leading-5 text-ink/55">{JSON.stringify(memory.value)}</p></div>)}{!memories.data?.length && <p className="text-xs text-ink/40">暂无已启用的长期记忆。</p>}</div></Card>
        </div>
      </div>
    </div>
  );
}
