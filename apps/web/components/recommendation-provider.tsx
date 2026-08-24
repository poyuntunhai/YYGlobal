"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createContext, useContext, useEffect, useState } from "react";
import { api, Profile, ProgramRecommendation, RecommendationGoal, RecommendationGoalInput } from "@/lib/api";

const STORAGE_KEY = "yyglobal-program-recommendations-v1";

type StoredSession = {
  profileVersion: string;
  query: string;
  items: ProgramRecommendation[];
  latestBatchIds: string[];
  selectedIds: string[];
};

type BatchRequest = {
  query: string;
  excludedIds: string[];
  profileVersion: string;
  replace: boolean;
};

type RecommendationContextValue = {
  profile?: Profile;
  goal?: RecommendationGoal;
  profileReady: boolean;
  input: string;
  setInput: (value: string) => void;
  queryText: string;
  items: ProgramRecommendation[];
  latestBatchIds: string[];
  selected: string[];
  setSelected: React.Dispatch<React.SetStateAction<string[]>>;
  recommendationNotice: string;
  isPending: boolean;
  error: Error | null;
  requestBatch: (replace: boolean, query?: string) => void;
  saveGoalAndAppend: (goal: RecommendationGoalInput) => Promise<void>;
  clearResults: () => void;
};

const RecommendationContext = createContext<RecommendationContextValue | null>(null);

function recommendationProfileVersion(profile?: Profile, goal?: RecommendationGoal) {
  if (!profile || !goal) return "";
  return JSON.stringify({
    current_major: profile.current_major,
    degree: profile.degree,
    gpa: profile.gpa,
    gpa_scale: profile.gpa_scale,
    language_scores: profile.language_scores,
    preferences: profile.preferences,
    confirmed: profile.confirmed,
    experiences: profile.experiences.map((item) => ({
      id: item.id,
      kind: item.kind,
      title: item.title,
      organization: item.organization,
      description: item.description,
      tags: item.tags,
      confirmed: item.confirmed,
    })),
    goal: {
      target_countries: goal.target_countries,
      target_fields: goal.target_fields,
      target_university: goal.target_university,
      target_degree_level: goal.target_degree_level,
      intake: goal.intake,
      budget: goal.budget,
      max_qs_rank: goal.max_qs_rank,
    },
  });
}

export function RecommendationProvider({ children }: { children: React.ReactNode }) {
  const client = useQueryClient();
  const [hydrated, setHydrated] = useState(false);
  const [input, setInput] = useState("");
  const [queryText, setQueryText] = useState("");
  const [items, setItems] = useState<ProgramRecommendation[]>([]);
  const [latestBatchIds, setLatestBatchIds] = useState<string[]>([]);
  const [sessionProfileVersion, setSessionProfileVersion] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [recommendationNotice, setRecommendationNotice] = useState("");
  const profileQuery = useQuery({ queryKey: ["profile"], queryFn: api.profile });
  const goalQuery = useQuery({ queryKey: ["recommendation-goal"], queryFn: api.recommendationGoal });
  const profile = profileQuery.data;
  const goal = goalQuery.data;
  const currentProfileVersion = recommendationProfileVersion(profile, goal);
  const profileReady = Boolean(
    profile?.confirmed
    && goal?.target_fields.length
    && (goal?.target_countries.length || goal?.target_university),
  );

  const fetchBatch = useMutation({
    mutationFn: (request: BatchRequest) => api.programRecommendations(request.query, request.excludedIds),
    onMutate: () => setRecommendationNotice(""),
    onSuccess: (batch, request) => {
      const freshIds = new Set(batch.map((item) => item.program.id));
      setItems((current) => request.replace ? batch : [...batch, ...current.filter((item) => !freshIds.has(item.program.id))]);
      setLatestBatchIds(batch.map((item) => item.program.id));
      setSessionProfileVersion(request.profileVersion);
      setQueryText(request.query);
      setInput(request.query);
      setRecommendationNotice(batch.length === 5
        ? `已联网发现并完成官网核验，在页面顶部新增 ${batch.length} 个项目。`
        : batch.length
          ? `本次联网搜索新增 ${batch.length} 个通过官网核验的项目。`
          : "本次已联网搜索，但没有发现新的、同时满足条件并通过官网证据核验的项目。可调整学校、QS、国家或专业方向后重试。");
    },
    onError: (error) => setRecommendationNotice(error instanceof Error ? error.message : "推荐失败"),
  });

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const stored = JSON.parse(raw) as Partial<StoredSession>;
        setItems(Array.isArray(stored.items) ? stored.items : []);
        setLatestBatchIds(Array.isArray(stored.latestBatchIds) ? stored.latestBatchIds : []);
        setSelected(Array.isArray(stored.selectedIds) ? stored.selectedIds : []);
        setSessionProfileVersion(stored.profileVersion ?? "");
        setQueryText(stored.query ?? "");
        setInput(stored.query ?? "");
      }
    } catch {
      window.localStorage.removeItem(STORAGE_KEY);
    }
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({
      profileVersion: sessionProfileVersion,
      query: queryText,
      items,
      latestBatchIds,
      selectedIds: selected,
    } satisfies StoredSession));
  }, [hydrated, items, latestBatchIds, queryText, selected, sessionProfileVersion]);

  const requestBatch = (replace: boolean, query = queryText) => {
    const profileVersion = currentProfileVersion || sessionProfileVersion;
    if (!profileVersion || fetchBatch.isPending) return;
    fetchBatch.mutate({ query, excludedIds: replace ? [] : items.map((item) => item.program.id), profileVersion, replace });
  };

  const saveGoalAndAppend = async (values: RecommendationGoalInput) => {
    if (fetchBatch.isPending) return;
    setRecommendationNotice("");
    try {
      const saved = await api.saveRecommendationGoal(values);
      client.setQueryData(["recommendation-goal"], saved);
      client.invalidateQueries({ queryKey: ["profile"] });
      const profileVersion = recommendationProfileVersion(profile, saved);
      if (!profileVersion) return;
      setSelected([]);
      fetchBatch.mutate({
        query: "",
        excludedIds: items.map((item) => item.program.id),
        profileVersion,
        replace: false,
      });
    } catch (error) {
      setRecommendationNotice(error instanceof Error ? error.message : "选校目标保存失败");
    }
  };

  const clearResults = () => {
    if (fetchBatch.isPending) return;
    setItems([]);
    setLatestBatchIds([]);
    setSelected([]);
    setQueryText("");
    setInput("");
    setSessionProfileVersion(currentProfileVersion);
    setRecommendationNotice("已清空全部推荐结果。调整条件后可继续推荐 5 个项目。");
  };

  return <RecommendationContext.Provider value={{
    profile, goal, profileReady, input, setInput, queryText, items, latestBatchIds,
    selected, setSelected, recommendationNotice, isPending: fetchBatch.isPending,
    error: fetchBatch.error instanceof Error ? fetchBatch.error : null,
    requestBatch, saveGoalAndAppend, clearResults,
  }}>{children}</RecommendationContext.Provider>;
}

export function useRecommendations() {
  const context = useContext(RecommendationContext);
  if (!context) throw new Error("useRecommendations must be used within RecommendationProvider");
  return context;
}
