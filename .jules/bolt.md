## 2024-05-23 - Lazy Loading React Components
**Learning:** React `lazy` and `Suspense` are effective for splitting code and reducing initial bundle size, even for moderately sized components like Settings pages or Modals.
**Action:** Identify other large components that are not immediately visible (e.g., behind tabs or modals) and lazy load them.

## 2024-05-24 - Optimized IPC Payload for ASR Streaming
**Learning:** Sending raw `int16` bytes over IPC to the worker process is significantly more efficient than sending pickled `float32` numpy arrays, reducing payload size by 50% and avoiding main-thread serialization overhead.
**Action:** When working with multiprocessing and audio data, prefer passing raw bytes and performing heavy conversions (like normalization) inside the worker process.
