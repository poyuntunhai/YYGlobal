"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, ChevronDown, ExternalLink, Plus, RefreshCw, Trash2, X } from "lucide-react";
import { KeyboardEvent, useEffect, useState } from "react";
import { PageHeader } from "@/components/page-header";
import { ProcessGuide } from "@/components/process-guide";
import { useRecommendations } from "@/components/recommendation-provider";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { api, RecommendationGoalInput } from "@/lib/api";
import { cn } from "@/lib/utils";

const countryOptions = [
  { value: "United States", label: "美国" },
  { value: "United Kingdom", label: "英国" },
  { value: "Canada", label: "加拿大" },
  { value: "Australia", label: "澳大利亚" },
  { value: "New Zealand", label: "新西兰" },
  { value: "Singapore", label: "新加坡" },
  { value: "Ireland", label: "爱尔兰" },
  { value: "Hong Kong", label: "中国香港（Hong Kong）" },
];

const emptyGoal: RecommendationGoalInput = {
  target_countries: [], target_fields: [], target_university: "", target_degree_level: "master", intake: "", budget: null, max_qs_rank: 100,
};

function universityKey(value: string) {
  const key = value
    .toLowerCase()
    .replace(/\([^)]*\)/g, " ")
    .replace(/[^a-z0-9]+/g, " ")
    .split(" ")
    .filter((token) => token && token !== "the")
    .join(" ");
  return ({
    ucl: "university college london",
    nus: "national university of singapore",
    ntu: "nanyang technological university",
  } as Record<string, string>)[key] ?? key;
}

function MultiSelectField({
  label, values, options, onChange,
}: {
  label: string;
  values: string[];
  options: Array<{ value: string; label: string }>;
  onChange: (values: string[]) => void;
}) {
  const optionLabels = new Map(options.map((item) => [item.value, item.label]));
  const toggle = (value: string) => onChange(
    values.includes(value) ? values.filter((item) => item !== value) : [...values, value],
  );

  return <label className="block min-w-0">
    <span className="label">{label}</span>
    <details className="group relative">
      <summary className="field flex min-h-11 cursor-pointer list-none items-center justify-between gap-3 [&::-webkit-details-marker]:hidden">
        <span className={values.length ? "font-bold text-ink" : "text-ink/40"}>{values.length ? `已选择 ${values.length} 项` : "请选择"}</span>
        <ChevronDown className="shrink-0 text-ink/40 transition group-open:rotate-180" size={16} />
      </summary>
      <div className="absolute left-0 z-30 mt-2 grid max-h-[min(24rem,60vh)] w-full min-w-0 grid-cols-1 gap-1 overflow-y-auto overscroll-contain rounded-lg border border-black/10 bg-white p-2 shadow-xl sm:w-max sm:min-w-[24rem] sm:grid-cols-2">
        {options.map((option) => <button key={option.value} type="button" onClick={() => toggle(option.value)} className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm hover:bg-paper">
          <span className={cn("grid size-5 shrink-0 place-items-center rounded border", values.includes(option.value) ? "border-moss bg-moss text-white" : "border-black/15")}>{values.includes(option.value) && <Check size={13} />}</span>
          {option.label}
        </button>)}
      </div>
    </details>
    {!!values.length && <div className="mt-2 flex flex-wrap gap-1.5">{values.map((value) => <span key={value} className="inline-flex max-w-full items-center gap-1 rounded-full bg-mint/70 px-2.5 py-1 text-xs font-bold text-moss"><span className="truncate">{optionLabels.get(value) ?? value}</span><button type="button" onClick={() => toggle(value)} aria-label={`移除${optionLabels.get(value) ?? value}`}><X size={12} /></button></span>)}</div>}
  </label>;
}

function DirectionInput({ values, onChange }: { values: string[]; onChange: (values: string[]) => void }) {
  const [inputValue, setInputValue] = useState("");
  const addValues = (rawValue: string) => {
    const additions = rawValue
      .split(/[,，;；\n]+/)
      .map((value) => value.trim())
      .filter(Boolean);
    if (!additions.length) return;
    onChange(Array.from(new Set([...values, ...additions])));
    setInputValue("");
  };
  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter" || event.key === "," || event.key === "，") {
      event.preventDefault();
      addValues(inputValue);
    }
  };

  return <label className="block min-w-0">
    <span className="label">目标方向（可填写多个）</span>
    <div className="flex min-h-11 items-center gap-2 rounded-lg border border-black/10 bg-white px-3 py-1.5 focus-within:border-moss/40 focus-within:ring-2 focus-within:ring-moss/10">
      <input
        className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-ink/35"
        value={inputValue}
        onChange={(event) => setInputValue(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="输入专业方向，按回车或逗号添加"
      />
      <button
        type="button"
        className="grid size-8 shrink-0 place-items-center rounded-md bg-paper text-moss transition hover:bg-mint disabled:cursor-not-allowed disabled:opacity-40"
        onMouseDown={(event) => event.preventDefault()}
        onClick={() => addValues(inputValue)}
        disabled={!inputValue.trim()}
        aria-label="添加目标方向"
      >
        <Plus size={15} />
      </button>
    </div>
    <p className="mt-1.5 text-xs text-ink/40">可输入任意专业名称，中英文均可。</p>
    {!!values.length && <div className="mt-2 flex flex-wrap gap-1.5">{values.map((value) => <span key={value} className="inline-flex max-w-full items-center gap-1 rounded-full bg-mint/70 px-2.5 py-1 text-xs font-bold text-moss"><span className="truncate">{value}</span><button type="button" onClick={() => onChange(values.filter((item) => item !== value))} aria-label={`移除${value}`}><X size={12} /></button></span>)}</div>}
  </label>;
}

export default function ProgramsPage() {
  const client = useQueryClient();
  const {
    profile, goal, items, latestBatchIds,
    selected, setSelected, recommendationNotice, isPending, error,
    saveGoalAndAppend, clearResults,
  } = useRecommendations();
  const [goalForm, setGoalForm] = useState<RecommendationGoalInput>(emptyGoal);
  const [shortlistNotice, setShortlistNotice] = useState("");
  const [clearConfirmOpen, setClearConfirmOpen] = useState(false);
  const [recommendationNoticeDismissed, setRecommendationNoticeDismissed] = useState(false);
  const shortlists = useQuery({ queryKey: ["shortlists"], queryFn: api.shortlists });
  const universityCountries = goalForm.target_countries;
  const universityRankLimit = goalForm.max_qs_rank;
  const universities = useQuery({
    queryKey: ["university-options", universityCountries, universityRankLimit],
    queryFn: () => api.universityOptions(universityCountries, universityRankLimit),
  });
  const shortlist = shortlists.data?.[0];
  const joined = new Set(shortlist?.items.map((item) => item.program.id) ?? []);
  const latest = new Set(latestBatchIds);
  useEffect(() => {
    if (goal) setGoalForm({
      target_countries: goal.target_countries,
      target_fields: goal.target_fields,
      target_university: goal.target_university,
      target_degree_level: goal.target_degree_level,
      intake: goal.intake,
      budget: goal.budget,
      max_qs_rank: goal.max_qs_rank,
    });
  }, [goal]);
  useEffect(() => {
    setRecommendationNoticeDismissed(false);
  }, [recommendationNotice]);
  const noMorePrograms = recommendationNotice.startsWith(
    "本次已联网搜索，但没有发现新的",
  );
  const normalizedSchool = universityKey(goalForm.target_university);
  const selectedSchool = universities.data?.find(
    (item) => universityKey(item.university) === normalizedSchool,
  );
  const schoolInputValid = !goalForm.target_university.trim() || Boolean(selectedSchool);
  const goalReady = Boolean(
    goalForm.target_fields.length
    && (goalForm.target_countries.length || selectedSchool)
    && schoolInputValid,
  );
  const goalDirty = Boolean(goal && JSON.stringify(goalForm) !== JSON.stringify({
    target_countries: goal.target_countries,
    target_fields: goal.target_fields,
    target_university: goal.target_university,
    target_degree_level: goal.target_degree_level,
    intake: goal.intake,
    budget: goal.budget,
    max_qs_rank: goal.max_qs_rank,
  }));
  const add = useMutation({
    mutationFn: (ids: string[]) => api.addShortlistItems(ids),
    onSuccess: () => {
      setSelected([]);
      setShortlistNotice("已加入选校清单。推荐任务仍在继续。");
      client.invalidateQueries({ queryKey: ["shortlists"] });
      client.invalidateQueries({ queryKey: ["application-packages"] });
    },
    onError: (error) => setShortlistNotice(error instanceof Error ? error.message : "加入失败"),
  });
  const remove = useMutation({
    mutationFn: ({ shortlistId, programId }: { shortlistId: string; programId: string }) => api.removeShortlistItem(shortlistId, programId),
    onSuccess: () => {
      setShortlistNotice("已从选校清单移除。推荐任务仍在继续。");
      client.invalidateQueries({ queryKey: ["shortlists"] });
      client.invalidateQueries({ queryKey: ["application-packages"] });
    },
    onError: (error) => setShortlistNotice(error instanceof Error ? error.message : "移除失败"),
  });

  const toggle = (id: string) => setSelected((current) => current.includes(id) ? current.filter((item) => item !== id) : [...current, id]);

  return <div className="mx-auto max-w-7xl">
    <PageHeader
      eyebrow="Profile-grounded recommendations"
      title="项目推荐"
      description="先确定本次选校目标，再结合画像与已确认经历检索项目，并核验官网要求和原文证据。"
      actions={<Button onClick={() => add.mutate(selected)} disabled={!selected.length || add.isPending}>加入选校清单（{selected.length}）</Button>}
    />
    <ProcessGuide current={1} completed={[0]} />
    {shortlistNotice && <div className="mb-5 rounded-xl bg-mint/70 px-4 py-3 text-sm text-moss">{shortlistNotice}</div>}
    {recommendationNotice && !noMorePrograms && <div className="mb-5 rounded-xl bg-sky-50 px-4 py-3 text-sm text-sky-800">{recommendationNotice}</div>}
    <Card className="mb-6">
      <div className="mb-5 flex items-end justify-between gap-4 border-b border-black/5 pb-4"><div><p className="eyebrow">Selection goal</p><h2 className="mt-1 text-lg font-black">本次选校目标</h2></div><span className="text-xs text-ink/45">筛选条件随检索保存</span></div>
      <form onSubmit={(event) => { event.preventDefault(); if (goalReady) void saveGoalAndAppend(goalForm); }}>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <MultiSelectField label="目标国家（多选）" values={goalForm.target_countries} options={countryOptions} onChange={(target_countries) => setGoalForm((current) => ({ ...current, target_countries, target_university: "" }))} />
          <label><span className="label">QS 2026 排名范围</span><select className="field" value={universityRankLimit ?? ""} onChange={(event) => setGoalForm((current) => ({ ...current, max_qs_rank: event.target.value ? Number(event.target.value) : null, target_university: "" }))}><option value="50">前 50</option><option value="100">前 100（默认）</option><option value="200">前 200</option><option value="300">前 300</option><option value="">不限 QS</option></select></label>
          <DirectionInput values={goalForm.target_fields} onChange={(target_fields) => setGoalForm((current) => ({ ...current, target_fields }))} />
          <label><span className="label">目标学位</span><select className="field" value={goalForm.target_degree_level} onChange={(event) => setGoalForm((current) => ({ ...current, target_degree_level: event.target.value as RecommendationGoalInput["target_degree_level"] }))}><option value="undergraduate">本科</option><option value="master">Master / 硕士</option><option value="doctoral">PhD / 博士</option></select></label>
          <label className="block min-w-0">
            <span className="label">指定学校（可搜索）</span>
            <input
              className="field"
              list="university-options"
              value={goalForm.target_university}
              onChange={(event) => setGoalForm((current) => ({ ...current, target_university: event.target.value }))}
              placeholder="输入学校名称进行搜索"
            />
            <datalist id="university-options">{universities.data?.map((item) => <option key={item.university} value={item.university}>QS {item.qs_rank} · {item.country} · {item.city}</option>)}</datalist>
            {selectedSchool && <p className="mt-1.5 text-xs text-ink/40">QS 2026 第 {selectedSchool.qs_rank} · {selectedSchool.country} · {selectedSchool.city}</p>}
            {!goalForm.target_university.trim() && <p className="mt-1.5 text-xs text-ink/40">当前范围内 {universities.data?.length ?? 0} 所学校，可输入名称搜索。不限 QS 时展示当前排名库收录的学校。</p>}
            {goalForm.target_university.trim() && !selectedSchool && <p className="mt-1.5 text-xs font-bold text-amber-700">请从当前国家与 QS 范围内的学校中选择。</p>}
          </label>
          <label><span className="label">入学时间</span><input className="field" value={goalForm.intake} onChange={(event) => setGoalForm((current) => ({ ...current, intake: event.target.value }))} placeholder="例如 2027 Fall" /></label>
          <label><span className="label">总预算（USD）</span><input className="field" type="number" min="0" value={goalForm.budget ?? ""} onChange={(event) => setGoalForm((current) => ({ ...current, budget: event.target.value ? Number(event.target.value) : null }))} placeholder="可选" /></label>
        </div>
        <div className="mt-5 flex flex-wrap items-center justify-between gap-3 border-t border-black/5 pt-4">
          <p className="text-xs text-ink/45">{goalDirty ? "筛选条件已修改，将在下一批推荐中生效。" : `当前按 ${goal?.target_fields.join("、") || "选校目标"} 匹配`}</p>
          <div className="flex flex-wrap gap-2"><Button type="button" variant="danger" onClick={() => setClearConfirmOpen(true)} disabled={isPending || !items.length}><Trash2 size={15} />清空全部结果</Button><Button type="submit" disabled={isPending || !goalReady}><Plus size={15} />再推荐 5 个</Button></div>
        </div>
      </form>
    </Card>
    {!profile?.confirmed && <Card className="mb-6 border-amber-200 bg-amber-50 text-center"><h2 className="text-lg font-black text-amber-900">请先确认个人画像</h2><p className="mt-2 text-sm text-amber-800">成绩、语言和真实经历是计算项目匹配度的基础。</p><a className="mt-4 inline-flex text-sm font-black text-amber-900 underline" href="/profile">前往填写画像</a></Card>}
    {profile?.confirmed && !goalReady && <Card className="mb-6 border-amber-200 bg-amber-50 text-center"><h2 className="text-lg font-black text-amber-900">请完善检索条件</h2><p className="mt-2 text-sm text-amber-800">至少填写一个专业方向，并选择目标国家或指定学校。</p></Card>}
    {isPending && <Card className="mb-6 py-12 text-center"><RefreshCw className="mx-auto animate-spin text-moss" /><p className="mt-4 font-black">正在检索并核验新一批 5 个项目官网…</p><p className="mt-2 text-sm text-ink/45">切换到其他页面不会中断推荐，完成后会保留结果。</p></Card>}
    {error && <Card className="mb-6 border-red-200 bg-red-50 text-sm text-red-800">{error.message}</Card>}
    <div className="grid gap-4 xl:grid-cols-2">
      {items.map(({ program, score, reasons, verification_status, verification_error }) => {
        const active = selected.includes(program.id);
        const isJoined = joined.has(program.id);
        const isLatest = latest.has(program.id);
        const requirement = program.requirement;
        return <Card key={program.id} className={cn("relative transition", (active || isJoined) && "border-moss/40 ring-2 ring-moss/10", isLatest && "border-sky-300")}>
          {isLatest && <span className="absolute right-4 top-4 rounded-full bg-sky-100 px-2.5 py-1 text-[11px] font-black text-sky-700">新推荐</span>}
          <div className="flex items-start gap-4 pr-16">
            {!isJoined && <button type="button" onClick={() => toggle(program.id)} className={cn("mt-1 grid size-6 shrink-0 place-items-center rounded-lg border transition", active ? "border-moss bg-moss text-white" : "border-black/15 bg-white")} aria-label={active ? "取消选择" : "选择项目"}>{active && <Check size={14} />}</button>}
            <div className="min-w-0 flex-1"><p className="eyebrow">{program.country} · {program.city}</p><h2 className="mt-2 text-lg font-black leading-6">{program.university}</h2><p className="mt-1 text-sm text-ink/60">{program.name}</p><span className="mt-2 inline-flex rounded-full bg-sky-50 px-2.5 py-1 text-[11px] font-bold text-sky-700">{program.field} · {program.degree}</span></div>
            <div className="text-right"><strong className="text-2xl text-moss">{score.toFixed(0)}</strong><p className="text-[10px] font-bold text-ink/40">匹配分</p></div>
          </div>
          <div className="mt-4 flex flex-wrap gap-2">{reasons.slice(0, 5).map((reason) => <span key={reason} className="rounded-full bg-mint/60 px-2.5 py-1 text-xs text-moss">{reason}</span>)}</div>
          {verification_status !== "verified" && <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs leading-5 text-amber-900">{verification_status === "failed" ? "本次官网读取失败，当前项目尚未完成官网核验。" : "官网已读取，但材料与截止日期证据尚不完整。"}{verification_error && <span className="ml-1 text-amber-700">{verification_error}</span>}</div>}
          <div className="mt-5 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">{[
            ["QS排名", program.qs_rank ? `${program.qs_rank}（${program.qs_ranking_year}）` : "暂无排名"],
            ["学费", program.tuition ? `${program.currency} ${program.tuition.toLocaleString()}` : "官网未列出"],
            ["截止", requirement?.deadline ?? "官网未列出"],
            ["GPA", requirement?.min_gpa ? `≥ ${requirement.min_gpa}` : "官网未列出"],
            ["语言", requirement?.language.TOEFL ? `TOEFL ${requirement.language.TOEFL}` : requirement?.language.IELTS ? `IELTS ${requirement.language.IELTS}` : "官网未列出"],
          ].map(([label, value]) => <div key={label} className="rounded-xl bg-paper/80 p-3"><p className="text-[11px] font-bold uppercase tracking-wider text-ink/40">{label}</p><p className="mt-1 text-sm font-bold">{value}</p></div>)}</div>
          {!!requirement?.materials.length && <div className="mt-4"><p className="label">申请材料</p><div className="mt-2 flex flex-wrap gap-2">{requirement.materials.map((item) => <span key={item} className="rounded-full bg-paper px-2.5 py-1 text-xs text-ink/65">{item}</span>)}</div></div>}
          {!!requirement?.prerequisites.length && <div className="mt-4"><p className="label">背景要求</p><p className="mt-1 text-sm leading-6 text-ink/60">{requirement.prerequisites.join("、")}</p></div>}
          {program.evidence.length > 0 && <details className="mt-4 rounded-xl border border-emerald-100 bg-emerald-50/60 p-3" open><summary className="cursor-pointer text-xs font-black text-emerald-800">官网原文证据（{program.evidence.length}）</summary><div className="mt-3 space-y-2">{program.evidence.slice(0, 8).map((item) => <blockquote key={item.id} className="border-l-2 border-emerald-300 pl-3 text-xs leading-5 text-ink/60"><span className="font-black text-emerald-800">{item.field}</span>：{item.quote}</blockquote>)}</div></details>}
          <div className="mt-5 flex flex-wrap items-center justify-between gap-3 border-t border-black/5 pt-4">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
              <a href={program.official_url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1.5 text-xs font-bold text-moss hover:underline">项目官网 <ExternalLink size={13} /></a>
              {program.catalog_url && <a href={program.catalog_url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1.5 text-xs font-bold text-sky-700 hover:underline">学校研究生项目目录 <ExternalLink size={13} /></a>}
              {program.faculty_catalog_url && <a href={program.faculty_catalog_url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1.5 text-xs font-bold text-violet-700 hover:underline">所属学院项目目录 <ExternalLink size={13} /></a>}
            </div>
            {isJoined && shortlist ? <Button variant="ghost" size="sm" onClick={() => remove.mutate({ shortlistId: shortlist.id, programId: program.id })} disabled={remove.isPending}><Trash2 size={13} />已加入 · 移除</Button> : <Button size="sm" variant="secondary" onClick={() => add.mutate([program.id])} disabled={add.isPending}>加入选校</Button>}
          </div>
        </Card>;
      })}
    </div>

    {clearConfirmOpen && <div className="fixed inset-0 z-50 grid place-items-center bg-ink/45 p-4" role="dialog" aria-modal="true" aria-labelledby="clear-results-title">
      <Card className="w-full max-w-md border-white bg-white shadow-2xl">
        <p className="eyebrow">Clear recommendations</p>
        <h2 id="clear-results-title" className="mt-2 text-xl font-black">清空全部推荐结果？</h2>
        <p className="mt-3 text-sm leading-6 text-ink/60">只会清空当前推荐页面中的结果，不会删除选校清单或申请包。</p>
        <div className="mt-6 flex justify-end gap-2"><Button variant="secondary" onClick={() => setClearConfirmOpen(false)}>取消</Button><Button variant="danger" onClick={() => { clearResults(); setClearConfirmOpen(false); }}>确认清空</Button></div>
      </Card>
    </div>}
    {noMorePrograms && !recommendationNoticeDismissed && <div className="fixed inset-0 z-50 grid place-items-center bg-ink/50 p-4" role="dialog" aria-modal="true" aria-labelledby="no-more-programs-title">
      <Card className="w-full max-w-lg border-white bg-white px-7 py-8 text-center shadow-2xl sm:px-10 sm:py-10">
        <div className="mx-auto grid size-14 place-items-center rounded-full bg-sky-50 text-sky-700"><RefreshCw size={24} /></div>
        <p className="eyebrow mt-5">Search completed</p>
        <h2 id="no-more-programs-title" className="mt-2 text-2xl font-black">没有更多符合条件的项目</h2>
        <p className="mt-4 text-sm leading-7 text-ink/60">{recommendationNotice}</p>
        <Button className="mt-7 min-w-32" onClick={() => setRecommendationNoticeDismissed(true)}>知道了</Button>
      </Card>
    </div>}
  </div>;
}
