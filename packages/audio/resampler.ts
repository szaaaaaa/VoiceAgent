/**
 * Audio channel downmixer, anti-aliased resampler to 16kHz mono PCM16 (FR-03),
 * and RMS dBFS meter.
 */

export const TARGET_ASR_SAMPLE_RATE = 16000;

export function float32ToInt16(samples: Float32Array): Int16Array {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]!));
    out[i] = clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767);
  }
  return out;
}

export function int16ToFloat32(samples: Int16Array): Float32Array {
  const out = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) {
    out[i] = samples[i]! / 32768;
  }
  return out;
}

export function downmixToMono(interleaved: Float32Array, channels: number): Float32Array {
  if (channels <= 1) return interleaved;
  const frames = Math.floor(interleaved.length / channels);
  const mono = new Float32Array(frames);
  const inv = 1 / channels;
  for (let i = 0; i < frames; i += 1) {
    let sum = 0;
    const base = i * channels;
    for (let c = 0; c < channels; c += 1) {
      sum += interleaved[base + c]!;
    }
    mono[i] = sum * inv;
  }
  return mono;
}

/**
 * Resamples mono Float32 PCM to targetSampleRate using area-averaging when downsampling
 * (acts as a box anti-aliasing filter) and linear interpolation when upsampling.
 */
export function resampleMonoFloat32(
  input: Float32Array,
  sourceSampleRate: number,
  targetSampleRate = TARGET_ASR_SAMPLE_RATE,
): Float32Array {
  if (input.length === 0 || sourceSampleRate === targetSampleRate) {
    return input;
  }
  if (sourceSampleRate <= 0 || targetSampleRate <= 0) {
    throw new Error(`Invalid sample rates: ${sourceSampleRate} -> ${targetSampleRate}`);
  }

  const ratio = sourceSampleRate / targetSampleRate;
  const outLen = Math.max(1, Math.round(input.length / ratio));
  const out = new Float32Array(outLen);

  if (ratio > 1) {
    // Downsampling: average all source samples contributing to each output bin.
    for (let i = 0; i < outLen; i += 1) {
      const start = Math.floor(i * ratio);
      const end = Math.min(input.length, Math.max(start + 1, Math.floor((i + 1) * ratio)));
      let sum = 0;
      for (let j = start; j < end; j += 1) sum += input[j]!;
      out[i] = sum / (end - start);
    }
  } else {
    // Upsampling: linear interpolation.
    for (let i = 0; i < outLen; i += 1) {
      const pos = i * ratio;
      const idx = Math.floor(pos);
      const frac = pos - idx;
      const a = input[Math.min(idx, input.length - 1)]!;
      const b = input[Math.min(idx + 1, input.length - 1)]!;
      out[i] = a + (b - a) * frac;
    }
  }
  return out;
}

export function toMono16kPcm16(
  interleavedFloat32: Float32Array,
  sourceSampleRate: number,
  channels = 1,
): Int16Array {
  const mono = downmixToMono(interleavedFloat32, channels);
  const resampled = resampleMonoFloat32(mono, sourceSampleRate, TARGET_ASR_SAMPLE_RATE);
  return float32ToInt16(resampled);
}

export function computeRmsDb(pcm16: Int16Array): number {
  if (pcm16.length === 0) return -120;
  let sumSq = 0;
  for (let i = 0; i < pcm16.length; i += 1) {
    const norm = pcm16[i]! / 32768;
    sumSq += norm * norm;
  }
  const rms = Math.sqrt(sumSq / pcm16.length);
  if (rms < 1e-6) return -120;
  return 20 * Math.log10(rms);
}
