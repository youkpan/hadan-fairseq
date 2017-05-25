-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Hypothesis generation script with text file input, processed line-by-line.
-- By default, this will run in interactive mode.
--
--]]

require 'fairseq'

local tnt = require 'torchnet'
local tds = require 'tds'
local argcheck = require 'argcheck'
local plstringx = require 'pl.stringx'
local data = require 'fairseq.torchnet.data'
local search = require 'fairseq.search'
local tokenizer = require 'fairseq.text.tokenizer'
local mutils = require 'fairseq.models.utils'

local cmd = torch.CmdLine()
cmd:option('-path', 'model1.th7,model2.th7', 'path to saved model(s)')
cmd:option('-path2', 'model1.th7,model2.th7', 'path to saved model(s)')
cmd:option('-beam', 1, 'search beam width')
cmd:option('-lenpen', 1,
    'length penalty: <1.0 favors shorter, >1.0 favors longer sentences')
cmd:option('-unkpen', 0,
    'unknown word penalty: <0 produces more, >0 produces less unknown words')
cmd:option('-subwordpen', 0,
    'subword penalty: <0 favors longer, >0 favors shorter words')
cmd:option('-covpen', 0,
    'coverage penalty: favor hypotheses that cover all source tokens')
cmd:option('-nbest', 1, 'number of candidate hypotheses')
cmd:option('-minlen', 1, 'minimum length of generated hypotheses')
cmd:option('-maxlen', 500, 'maximum length of generated hypotheses')
cmd:option('-input', '-', 'source language input text file')
cmd:option('-sourcedict', '', 'source language dictionary')
cmd:option('-targetdict', '', 'target language dictionary')
cmd:option('-vocab', '', 'restrict output to target vocab')
cmd:option('-visdom', '', 'visualize with visdom: (host:port)')
cmd:option('-model', '', 'model type for legacy models')
cmd:option('-aligndictpath', '', 'path to an alignment dictionary (optional)')
cmd:option('-nmostcommon', 500,
    'the number of most common words to keep when using alignment')
cmd:option('-topnalign', 100, 'the number of the most common alignments to use')
cmd:option('-freqthreshold', -1,
    'the minimum frequency for an alignment candidate in order' ..
    'to be considered (default no limit)')
cmd:option('-fconvfast', false, 'make fconv model faster')

local config = cmd:parse(arg)

-------------------------------------------------------------------
-- Load data
-------------------------------------------------------------------
config.dict = torch.load(config.targetdict)
print(string.format('| [target] Dictionary: %d types',  config.dict:size()))
config.srcdict = torch.load(config.sourcedict)
print(string.format('| [source] Dictionary: %d types',  config.srcdict:size()))

if config.aligndictpath ~= '' then
    config.aligndict = tnt.IndexedDatasetReader{
        indexfilename = config.aligndictpath .. '.idx',
        datafilename = config.aligndictpath .. '.bin',
        mmap = true,
        mmapidx = true,
    }
    config.nmostcommon = math.max(config.nmostcommon, config.dict.nspecial)
    config.nmostcommon = math.min(config.nmostcommon, config.dict:size())
end

local TextFileIterator, _ =
    torch.class('tnt.TextFileIterator', 'tnt.DatasetIterator', tnt)

TextFileIterator.__init = argcheck{
    {name='self', type='tnt.TextFileIterator'},
    {name='path', type='string'},
    {name='transform', type='function',
        default=function(sample) return sample end},
    call = function(self, path, transform)
        function self.run()
            local fd
            if path == '-' then
                fd = io.stdin
            else
                fd = io.open(path)
            end
            return function()
                if torch.isatty(fd) then
                    io.stdout:write('> ')
                    io.stdout:flush()
                end
                local line = fd:read()
                if line ~= nil then
                    return transform(line)
                elseif fd ~= io.stdin then
                    fd:close()
                end
            end
        end
    end
}

local dataset = tnt.DatasetIterator{
    iterator = tnt.TextFileIterator{
        path = config.input,
        transform = function(line)
            return {
                bin = tokenizer.tensorizeString(line, config.srcdict),
                text = line,
            }
        end
    },
    transform = function(sample)
        local source = sample.bin:view(-1, 1):int()
        local sourcePos = data.makePositions(source,
            config.srcdict:getPadIndex()):view(-1, 1)
        local sample = {
            source = source,
            sourcePos = sourcePos,
            text = sample.text,
            target = torch.IntTensor(1, 1), -- a stub
        }
        if config.aligndict then
            sample.targetVocab, sample.targetVocabMap,
                sample.targetVocabStats
                    = data.getTargetVocabFromAlignment{
                        dictsize = config.dict:size(),
                        unk = config.dict:getUnkIndex(),
                        aligndict = config.aligndict,
                        set = 'test',
                        source = sample.source,
                        target = sample.target,
                        nmostcommon = config.nmostcommon,
                        topnalign = config.topnalign,
                        freqthreshold = config.freqthreshold,
                    }
        end
        return sample
    end,
}

local model
if config.model ~= '' then
    model = mutils.loadLegacyModel(config.path, config.model)
else
    model = require(
        'fairseq.models.ensemble_model'
    ).new(config)
    if config.fconvfast then
        local nfconv = 0
        for _, fconv in ipairs(model.models) do
            if torch.typename(fconv) == 'FConvModel' then
                fconv:makeDecoderFast()
                nfconv = nfconv + 1
            end
        end
        assert(nfconv > 0, '-fconvfast requires an fconv model in the ensemble')
    end

    model2 = require(
        'fairseq.models.ensemble_model'
    ).new(config)

    --print(config)
    config.nhids = {}
    config.kwidths = {}
    
    --config.usemodel = 4
    print(config.model ,config.kwidths)
    config.attnlayers = 0

    model4 = require(
        'fairseq.models.ensemble_model'
    ).new(config)
end

local vocab = nil
if config.vocab ~= '' then
    vocab = tds.Hash()
    local fd = io.open(config.vocab)
    while true do
        local line = fd:read()
        if line == nil then
            break
        end
        -- Add word on this line together with all prefixes
        for i = 1, line:len() do
            vocab[line:sub(1, i)] = 1
        end
    end
end
local searchf = search.beam{
    ttype = model:type(),
    dict = config.dict,
    srcdict = config.srcdict,
    beam = config.beam,
    lenPenalty = config.lenpen,
    unkPenalty = config.unkpen,
    subwordPenalty = config.subwordpen,
    coveragePenalty = config.covpen,
    vocab = vocab,
}
local searchf_src = search.beam{
    ttype = model:type(),
    dict = config.dict,
    srcdict = config.srcdict,
    beam = config.beam,
    lenPenalty = config.lenpen,
    unkPenalty = config.unkpen,
    subwordPenalty = config.subwordpen,
    coveragePenalty = config.covpen,
    vocab = vocab,
}


if config.visdom ~= '' then
    local host, port = table.unpack(plstringx.split(config.visdom, ':'))
    searchf = search.visualize{
        sf = searchf,
        dict = config.dict,
        sourceDict = config.srcdict,
        host = host,
        port = tonumber(port),
    }
end

local dict, srcdict = config.dict, config.srcdict
local eos = dict:getSymbol(dict:getEosIndex())
local seos = srcdict:getSymbol(srcdict:getEosIndex())
local unk = dict:getSymbol(dict:getUnkIndex())

-- Select unknown token for reference that can't be produced by the model so
-- that the program output can be scored correctly.
local runk = unk
repeat
    runk = string.format('<%s>', runk)
until dict:getIndex(runk) == dict:getUnkIndex()

function md5_sumhexa(k)
    local md5_core = require "md5.core"
    k = md5_core.sum(k)
    return (string.gsub(k, ".", 
        function (c) 
            return string.format("%02x", string.byte(c)) 
        end
        )
    )
end
--print(md5_sumhexa("Hello World!"))

sentence_bufer={}
sentence_bufer_idx = 1
plp = require('pl.pretty')

for sample in dataset() do
    sample.bsz = 1
    
    local conv = model2:generate(config, sample, searchf ,2 ,torch.CudaTensor(1))
    --print("conv[1] size",conv[1]:size())
    for k,v in pairs(conv) do
        --print(k," conv:size",v:size(1),v:size(2),v:size(3))
    end

    -- Print results
    local sourceString = config.srcdict:getString(sample.source:t()[1])
    sourceString = sourceString:gsub(seos .. '.*', '')
    print('S', sourceString)

    --local testi = math.random()*1000
    --testi = math.floor(testi)
    testi = md5_sumhexa(sourceString):sub(1,6)
    image = require('image')
    word_a = sourceString:split(" ")
    sentence_v = {}

    sentence_len = conv[1]:size(1)
    print("sentence_len " ,sentence_len)
    sentence_v2 = torch.FloatTensor(256):fill(0.5)

    --table.insert(sentence_v,sentence_len_bar)

    if sentence_len >40 then
        sentence_len = 40
    end

    if sentence_len == nil then
        sentence_len = 0
        print('sentence_len nil')
        return
    end

    for i = 1 ,sentence_len  do
        conv_all = conv[1][i]
        words = word_a[i]
        local img =  torch.FloatTensor(conv_all:size()):copy(conv_all)
        --img=img:resize(conv_all:size(1),conv_all:size(2))
        img2 = img:resize(256)
        
        img2 = img2 - img2:min()
        img2 = img2 / img2:max()
        --table.insert(sentence_v,img2)
        --print(img2)
        sentence_v2 = sentence_v2:cat(img2)
        --print("sentence_v get")
    end
    --print(sentence_v2)
    --print(sentence_v2:size())
    --print(img2)
    --img2 = image.scale(sentence_v,16*40,16*40)
    --path1=string.format('/home/pan/fairseq/s_%s_p_%d_%s.jpg',testi,i,words)
    path1=string.format('/home/pan/fairseq/sentence/s_%d',sentence_bufer_idx)
    print(path1)
    sentence_bufer_idx = sentence_bufer_idx +1
    --[[

    if #sentence_bufer == 0 then
        sentence_bufer[1]=sentence_v
        sentence_bufer[2]=sentence_v
        sentence_bufer[3]=sentence_v
    end
    table.insert(sentence_v2,sentence_bufer[1])
    table.insert(sentence_v2,sentence_bufer[2])
    table.insert(sentence_v2,sentence_bufer[3])
    table.insert(sentence_v2,sentence_v)

    sentence_bufer[1] = sentence_bufer[2]
    sentence_bufer[2] = sentence_bufer[3]
    sentence_bufer[3] = sentence_v
    --]]

    sentence_v2:resize(1,sentence_len+1,256)
    --image.savePGM(path1,sentence_v2)

    --io.stdout:flush()
    --print(sample.source)
    local ts = sample.source
    local attnlayers = config.attnlayers
    local utils = require 'fairseq.utils'

    local hypos_s  = model4:generate(config, sample, searchf_src , 4 , conv[1])
    config.attnlayers =attnlayers
    --plp.dump(hypos_s)
    canshow = 1
    for i = 1, hypos_s:size(1) do
       if hypos_s[i] > config.srcdict:size() or  hypos_s[i] <=0  then
            canshow = 0
            hypos_s[i] = 1
        end
    end

    if canshow ==1 then
        sourceString2 = config.srcdict:getString(hypos_s):gsub(eos .. '.*', '')
        sourceString2 = sourceString2:gsub(seos .. '.*', '')
        sstr = ""
        for i=1,hypos_s:size(1) do
            tt= string.format("%d",hypos_s[i])
            sstr = sstr..tt.." "
        end
        print("I",sstr)
        print('B', sourceString2)
    end

    sample.source=ts

    local hypos, scores, attns = model:generate(config, sample, searchf , 0 ,torch.CudaTensor(1))
    local sstr=''
    --print(sample.source)

    
    for i=1,sample.source:size(1) do
        tt= string.format("%d",sample.source[i][1])
        sstr = sstr..tt.." "
    end
    print("I",sstr)
    
    print('O', sample.text)

    for i = 1, math.min(config.nbest, config.beam) do
        local hypo = config.dict:getString(hypos[i]):gsub(eos .. '.*', '')
        print('H', scores[i], hypo)
        -- NOTE: This will print #hypo + 1 attention maxima. The last one is the
        -- attention that was used to generate the <eos> symbol.
        local _, maxattns = torch.max(attns[i], 2)
        print('A', table.concat(maxattns:squeeze(2):totable(), ' '))
    end

    io.stdout:flush()
    
end
