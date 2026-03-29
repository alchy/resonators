#pragma once
/*
 * dsp_chain.h — stub matching IthacaCore DspChain interface.
 * Replace with the full IthacaCore dsp_chain.h + dsp_chain.cpp when
 * copying the DSP files from that project.
 */

class DspChain {
public:
    void prepare(int /*sample_rate*/, int /*max_block_size*/) {}
    void reset()  {}
    void process(float* /*L*/, float* /*R*/, int /*n*/) {}
    int  getEffectCount() const { return 0; }
};
