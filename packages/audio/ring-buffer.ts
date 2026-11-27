/**
 * Bounded circular Int16 PCM ring buffer & pre-roll buffer (FR-04).
 * Prevents unbounded memory growth during multi-hour continuous listening.
 */

export class Int16RingBuffer {
  private readonly buffer: Int16Array;
  private writePos = 0;
  private filled = 0;

  constructor(readonly capacitySamples: number) {
    if (!Number.isInteger(capacitySamples) || capacitySamples <= 0) {
      throw new Error(`Invalid ring buffer capacity: ${capacitySamples}`);
    }
    this.buffer = new Int16Array(capacitySamples);
  }

  get length(): number {
    return this.filled;
  }

  clear(): void {
    this.writePos = 0;
    this.filled = 0;
  }

  push(samples: Int16Array): void {
    const n = samples.length;
    if (n === 0) return;

    if (n >= this.capacitySamples) {
      // Keep only the newest capacitySamples samples.
      this.buffer.set(samples.subarray(n - this.capacitySamples), 0);
      this.writePos = 0;
      this.filled = this.capacitySamples;
      return;
    }

    const firstPart = Math.min(n, this.capacitySamples - this.writePos);
    this.buffer.set(samples.subarray(0, firstPart), this.writePos);
    if (firstPart < n) {
      this.buffer.set(samples.subarray(firstPart), 0);
    }
    this.writePos = (this.writePos + n) % this.capacitySamples;
    this.filled = Math.min(this.capacitySamples, this.filled + n);
  }

  readLatest(sampleCount = this.filled): Int16Array {
    const count = Math.min(Math.max(0, sampleCount), this.filled);
    const out = new Int16Array(count);
    if (count === 0) return out;

    const start = (this.writePos - count + this.capacitySamples) % this.capacitySamples;
    const firstPart = Math.min(count, this.capacitySamples - start);
    out.set(this.buffer.subarray(start, start + firstPart), 0);
    if (firstPart < count) {
      out.set(this.buffer.subarray(0, count - firstPart), firstPart);
    }
    return out;
  }
}

export class PreRollBuffer {
  private readonly ring: Int16RingBuffer;
  readonly prerollMs: number;
  readonly sampleRate: number;

  constructor(prerollMs = 300, sampleRate = 16000) {
    this.prerollMs = prerollMs;
    this.sampleRate = sampleRate;
    const samples = Math.max(1, Math.round((prerollMs / 1000) * sampleRate));
    this.ring = new Int16RingBuffer(samples);
  }

  push(samples: Int16Array): void {
    this.ring.push(samples);
  }

  snapshot(): Int16Array {
    return this.ring.readLatest();
  }

  clear(): void {
    this.ring.clear();
  }
}
