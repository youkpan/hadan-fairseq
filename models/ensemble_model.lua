-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Ensemble model.
-- Can only be used for inference.
--
--]]

require 'nn'
require 'rnnlib'
local argcheck = require 'argcheck'
local plstringx = require 'pl.stringx'
local utils = require 'fairseq.utils'

local cuda = utils.loadCuda()

local EnsembleModel = torch.class('EnsembleModel', 'Model')

EnsembleModel.__init = argcheck{
    {name='self', type='EnsembleModel'},
    {name='config', type='table'},
    call = function(self, config)

        if config.uesmodel == 4 then
            local paths = plstringx.split(config.path2, ',')
            self.models = {}
            for i, path in pairs(paths) do
                self.models[i] = torch.load(path)
            end
        else
            local paths = plstringx.split(config.path, ',')
            self.models = {}
            for i, path in pairs(paths) do
                self.models[i] = torch.load(path)
            end
        end
    end
}

EnsembleModel.type = argcheck{
    doc=[[
Shorthand for network():type()
]],
    {name='self', type='EnsembleModel'},
    {name='type', type='string', opt=true},
    {name='tensorCache', type='table', opt=true},
    call = function(self, type, tensorCache)
        local ret = nil
        for _, model in ipairs(self.models) do
            ret = model:type(type, tensorCache)
        end
        return not type and ret or self
    end
}

EnsembleModel.make = argcheck{
    {name='self', type='EnsembleModel'},
    {name='config', type='table'},
    call = function(self, config)
        error('Cannot construct a nn.Module instance for ensemble modules')
    end
}


EnsembleModel.generate = argcheck{
    {name='self', type='EnsembleModel'},
    {name='config', type='table'},
    {name='sample', type='table'},
    {name='search', type='table'},
    {name='typee', type='number'},
    {name='srcsoftmax', type='torch.CudaTensor'},
    call = function(self, config, sample, search ,typee ,srcsoftmax)
        local dict = config.dict
        local srcdict = config.srcdict
        local minlen = config.minlen
        local maxlen = config.maxlen
        local sourceLen = sample.source:size(1)
        local bsz = sample.source:size(2)
        local bbsz = config.beam * bsz

        local timers = {
            setup = torch.Timer(),
            encoder = torch.Timer(),
            decoder = torch.Timer(),
            search_prune = torch.Timer(),
            search_results = torch.Timer(),
        }

        local callbacks = {}
        for _, model in ipairs(self.models) do
            table.insert(callbacks, model:generationCallbacks(config, bsz  ))
        end


        for _, timer in pairs(timers) do
            timer:stop()
            timer:reset()
        end

        local states = {}
        for i = 1, #self.models do
            states[i] = {
                sample = sample,
            }
        end

        timers.setup:resume()
        local states = {}
        for i = 1, #self.models do
            states[i] = callbacks[i].setup(sample)
        end
        timers.setup:stop()

        timers.encoder:resume()
        for i = 1, #self.models do
            callbacks[i].encode(states[i])
        end
        if cuda.cutorch then
            cuda.cutorch.synchronize()
        end
        timers.encoder:stop()

        -- <eos> is used as a start-of-sentence marker
        local targetIns = {}
        for i = 1, #self.models do
            targetIns[i] = torch.Tensor(bbsz):type(self:type())
            targetIns[i]:fill(dict:getEosIndex())
            --print("targetIns[i]")
            --print(targetIns[i])
        end

        search.init(bsz, sample)
        local vocabsize =
            sample.targetVocab and sample.targetVocab:size(1) or dict:size()
        local srcvocabsize =  srcdict:size()   
        local aggSoftmax = torch.zeros(bbsz, vocabsize):type(self:type())
        local aggSoftmax_s = torch.zeros(bbsz, srcvocabsize):type(self:type())
        local aggAttnScores = torch.zeros(bbsz, sourceLen):type(self:type())
        local plp = require('pl.pretty')
        if typee == 4 then
            local m = self.models[1]:network()
            local mutils = require 'fairseq.models.utils'
            local encoder = mutils.findAnnotatedNode(m, 'encoder')
            local aggSoftmax = torch.zeros(sourceLen, srcvocabsize):type(self:type())
            --sourcePos={}
            --sample.source = {}
            --print(sample.source)
            --print("sourcePos")
            --print(sample.sourcePos)
            require('cutorch')
            findsize = 1000
            src_disc_map = nil -- config.src_disc_map
            if src_disc_map==nil then
                src_disc_map=torch.CudaDoubleTensor(0)
                config.src_disc_map2 = {}
                for k=1 ,20 do
                    config.src_disc_map2[k] = torch.CudaDoubleTensor(0)
                    --k pos ,word vector
                end

                local weights0 = encoder:get(2):parameters()[1]
                local weights = torch.CudaDoubleTensor(weights0:size()):copy(weights0)
                src_disc_map = weights:t()
                --[[
                for starti = 1, srcvocabsize,1000  do
                    if starti == 6001 then --6204
                        findsize = 204
                    end
                    local tmp = torch.range(starti, findsize+starti,1):resize(findsize,1)
                    source = torch.IntTensor(findsize,1):copy(tmp)
                    tmp = torch.ones(findsize):resize(findsize,1)
                    sourcePos = torch.IntTensor(findsize,1):copy(tmp)

                    --local tmp = (torch.ones(20)*starti):resize(20,1)
                    --source = torch.IntTensor(20,1):copy(tmp)
                    --tmp = torch.range(1, 20):resize(20,1)
                    --sourcePos = torch.IntTensor(20,1):copy(tmp)
                    if starti %1000 == 2 then
                        print ("getting word vector",starti)
                    end

                    src_disc_map2 = encoder:forward({source, sourcePos})
                    ----print(type(src_disc_map2[1]))
                    ----print(src_disc_map2[1]:size())
                    --print("src_disc_map2[1] size",src_disc_map2[1]:size())

                    src_disc_map = src_disc_map:cat(torch.CudaDoubleTensor(src_disc_map2[1]:size()):copy(src_disc_map2[1]),1)
                    --[[
                    for k=1 ,20 do
                        --src_disc_map2 = torch.CudaDoubleTensor(src_disc_map:size(2),src_disc_map:size(3)):copy(src_disc_map[k])
                        --src_disc_map2 = src_disc_map2:resize(src_disc_map:size(2),src_disc_map:size(3))
                        --print("src_disc_map2[1]k size",src_disc_map2[1][k]:size())
                        -- 1x256
                        local posvector = torch.CudaDoubleTensor(1,src_disc_map2[1]:size(3)):copy(src_disc_map2[1][k]) 
                        posvector = posvector:resize(src_disc_map2[1]:size(2),src_disc_map2[1]:size(3))--1x256
                        --posvector = posvector/3 --kwidth
                        config.src_disc_map2[k] = config.src_disc_map2[k]:cat(posvector,1)
                        --k pos ,word vector
                    end
                    ] ]
                    --print("src_disc_map:size",src_disc_map:size())
                end

                    src_disc_map = src_disc_map:resize(srcvocabsize,src_disc_map:size(3))
                    --6204x256 , to 
                    src_disc_map = src_disc_map:t()
                ]]

                --[[
                    for k=1 ,20 do
                        config.src_disc_map2[k] = config.src_disc_map2[k]:t()
                    end
                    --]]
                --config.src_disc_map = src_disc_map
                --config.src_disc_map = 1


            else
                --src_disc_map = src_disc_map:resize(srcvocabsize,srcsoftmax:size(3))
            end



            --plp.dump(src_disc_map)
            --print("bbsz",bbsz)
            
            --EnsembleModel
            local aggAttnScores2 = torch.CudaDoubleTensor(bbsz, sourceLen):type(self:type()):fill(0)
            local srcdict_2 = torch.CudaDoubleTensor(srcvocabsize):type(self:type())
            for j=1,srcvocabsize do
                ----print(srcdict[j])
                --srcdict_2[j]=encoderout[1][j]
            end
            --print("srcsoftmax type ",type(srcsoftmax))

            --srcsoftmax = src_v --[step]
            srcsoftmax0  = srcsoftmax
            --print(srcsoftmax:size())
            --srcsoftmax = torch.CudaDoubleTensor(srcsoftmax:size()):copy(srcsoftmax)
            srcsoftmax_v = {}
            for k=1 , srcsoftmax:size(1) do
                srcsoftmax_v[k] = torch.CudaDoubleTensor(srcsoftmax:size(2)):copy(srcsoftmax[k])
                srcsoftmax_v[k] = srcsoftmax_v[k]:resize(1,srcsoftmax:size(2))
            end

            --print(srcsoftmax_v[1])
            --srcsoftmax = srcsoftmax:resize(srcsoftmax:size(1),srcsoftmax:size(3))


            local pruned 
            local index_t =torch.CudaLongTensor(0)
            --predict_index = {}
            local aggAttnScores = torch.zeros(bbsz, sourceLen+1):type(self:type())
            for step = 1, sourceLen-1 do
                local srcsoftmax1
                if step == 1 then
                    srcsoftmax1 = srcsoftmax_v[1]
                    last_srcsoftmax = 0
                    last_srcsoftmax2 = 0
                elseif step == 2 then
                    srcsoftmax1 = srcsoftmax_v[2] -- srcsoftmax_v[1]
                else
                    srcsoftmax1 = srcsoftmax_v[step] -- 0.2*last_srcsoftmax - 0.2*last_srcsoftmax2
                end
                 
                last_srcsoftmax = last_srcsoftmax2
                last_srcsoftmax2 = srcsoftmax1

                aggAttnScores:zero()
                aggSoftmax:zero()
                idx = step
                if idx >sourceLen then
                    idx = sourceLen
                end
                aggAttnScores[1]=1.0

                --print("srcsoftmax size ",srcsoftmax1:size(),srcsoftmax1:type())
                --print("src_disc_map2 size ",config.src_disc_map2[step]:size())

                srcsoftmax2 = srcsoftmax1 * src_disc_map
                --print("srcsoftmax2 size ",srcsoftmax2:size())
                --print("aggSoftmax size ",aggSoftmax:size())
                local aggSoftmax2 = torch.zeros(1, srcvocabsize):type(self:type()):copy(srcsoftmax2)
                --aggSoftmax:add(srcsoftmax2)
                aggSoftmax2:div(#self.models)
                local aggLogSoftmax = aggSoftmax2:log()
                self:updateMinMaxLenProb(aggLogSoftmax, srcdict, 3, minlen, maxlen)

                max = srcsoftmax2[1][1]
                index = torch.CudaLongTensor(1):fill(srcsoftmax2[1][1])

                for k = 1 ,srcvocabsize do
                    if max < srcsoftmax2[1][k] then
                        max = srcsoftmax2[1][k]
                        index = torch.CudaLongTensor(1):fill(k)
                    end
                end

                --print("top i",index)
                --[[
                topScores,topIndices = search.prune2(sourceLen, aggLogSoftmax, aggAttnScores)
                --predict_index[step] = pruned.nextIn
                max = topScores[1][1]
                index = torch.CudaLongTensor(1):fill(topIndices[1][1])

                for k = 2 ,topScores:size(2) do
                    if max < topScores[1][k] then
                        max = topScores[1][k]
                        index = torch.CudaLongTensor(1):fill(topIndices[1][k])
                    end
                end
                --print("type topIndices",type(topIndices))
                plp.dump(topIndices)
                --print("topScores",max)
                --print("index",index)
                if index[1] > srcvocabsize or index[1] < 0 then
                    index = torch.CudaLongTensor(1):fill(1)
                end
                ]]

                index_t = index_t:cat(index)
                --table.insert(index_t,index)
            end
                --plp.dump(topScores)
                --plp.dump(index_t)

            return index_t

        end
        

        --print("generate conv ,#self.models",#self.models)
        --local conv = callbacks[3].decode2(states[3], targetIns[3])
        
        --plp.dump(conv)
        local conv_all
        image = require('image')
        --print(' start steps ')
        local conv1

        -- We do maxlen + 1 steps to give model a chance to predict EOS
        for step = 1, maxlen + 1 do
            timers.decoder:resume()
            aggSoftmax:zero()
            aggAttnScores:zero()

            local conv3
            local i
            for i = 1, #self.models do
                if typee == 5 then
                    local m = self.models[1]:network()
                    local mutils = require 'fairseq.models.utils'
                    local encoder = mutils.findAnnotatedNode(m, 'encoder')
                    local weights0 = encoder:get(2):parameters()[1]
                    local weights = torch.CudaTensor(weights0:size()):copy(weights0)

                    return weights
                end
                if typee == 2 then
                    --print("sample.source")
                    --print(sample.source)

                    local m = self.models[1]:network()
                    local mutils = require 'fairseq.models.utils'
                    local encoder = mutils.findAnnotatedNode(m, 'encoder')
                    --local j 
                    --plp.dump(encoder)
                    --print("encoder  size ",encoder:size())
                    --[[
                    params, gradParams = encoder:getParameters()
                    print("encoder size" ,params:size(1))

                    for j=1, encoder:size() do
                       local params = encoder:get(j):parameters()
                       if params then
                         local weights = params[1]
                         local biases  = params[2]
                         print("weights j ",j,"size",weights:size())
                         --n_parameters  = n_parameters + weights:nElement() + biases:nElement()
                       end
                    end

                    --print(encoder[1])
                    for k,v in pairs(encoder) do
                        --print("encoder ",k,v)
                    end
                    for j =  1, 4 do
                        --local p = encoder.get(encoder)
                        print("encoder ",j," size ",encoder[j]:size())
                    end 

                    ]]

                    local weights0 = encoder:get(2):parameters()[1]

                    local weights = torch.CudaTensor(weights0:size()):copy(weights0)
                    --print("weights  size",weights:size())

                    --plp.dump(weights[289])
                    weight_word=torch.CudaTensor()
                    for j =  1, sample.source:size(1) do
                        --print(sample.source[j])
                        --print('weights[sample.source[j]] size',weights[sample.source[j]]:size())
                       local s=sample.source[j]
                       --print(s[1])
                       local tweight = weights[s[1]] 
                       --print(tweight:size())
                       weight_word=weight_word:cat(tweight,2)
                    end 
                    weight_word = weight_word:t()
                    --print(weight_word:size())
                    return table.pack(weight_word)
                    --encoderout = encoder:forward({sample.source, sample.sourcePos})
                    --print('ok')
                    --return encoderout
                    
                    --conv1 = callbacks[i].decode(states[i], targetIns[i],2)
                    --return conv1
                end

                local softmax,conv = callbacks[i].decode(states[i], targetIns[i],1)
                conv3 = conv
                --print(conv)
                --print("-----type softmax",type(softmax))
                --print("-----size softmax",softmax:size())

                aggSoftmax:add(softmax)
                if callbacks[i].attention then
                    aggAttnScores:add(callbacks[i].attention(states[i]))
                end
            end
            
            --local img =  torch.FloatTensor(conv3:size()):copy(conv3)
            --img=img:resize(40,32)
            --img = img - img:min()
            --img = img / img:max()
            --img = image.scale(img,800,640)
            --path1=string.format('/home/pan/conv_step_%d.jpg',step)
            --image.saveJPG(path1,img)
            --]]--
            
            if step == 3 then
                conv_all = conv3
                --print('get conv_all')
            end
            
            -- Average softmax and attention scores.
            aggSoftmax:div(#self.models)
            aggAttnScores:div(#self.models)
            if cuda.cutorch then
                cuda.cutorch.synchronize()
            end
            timers.decoder:stop()

            local aggLogSoftmax = aggSoftmax:log()
            self:updateMinMaxLenProb(aggLogSoftmax, dict, step, minlen, maxlen)

            timers.search_prune:resume()

            local pruned = search.prune(step, aggLogSoftmax, aggAttnScores)

            timers.search_prune:stop()

            for i = 1, #self.models do
                targetIns[i]:copy(pruned.nextIn)
                callbacks[i].update(states[i], pruned.nextHid)
            end

            if pruned.eos then
                break
            end
        end
        ----
        --print(aggAttnScores:size())
        local t = torch.FloatTensor(aggAttnScores:size()):copy(aggAttnScores)
        local sum = 0
        for i=1 , aggAttnScores:size(1)  do
            sum = sum + t[i]
            if i%aggAttnScores:size(2)==(aggAttnScores:size(2)-1) then
                --print("aggAttnScores ",i,"  ",sum)
                sum = 0
            end
            
        end
        --plp.dump(conv_all)
        ---

        timers.search_results:resume()
        local results = table.pack(search.results())
        -- This is pretty hacky, but basically we can't run finalize for
        -- the selection models many times, because it will remap ids many times
        -- TODO: refactor this
        callbacks[1].finalize(states[1], sample, results)
        timers.search_results:stop()

        local times = {}
        for k, v in pairs(timers) do
            times[k] = v:time()
        end
        table.insert(results, times)
        table.insert(results, deepcopy1(conv_all))

        print("generate")
        --plp.dump(results[2])
        --print("inex")
        --plp.dump(results[4])
        --plp.dump(results)
        return table.unpack(results)
    end
}

function deepcopy1(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepcopy(orig_key)] = deepcopy1(orig_value)
        end
        setmetatable(copy, deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

function deepcopy(object)
    local lookup_table = {}
    local function _copy(object)
        if type(object) ~= "table" then
            return object
        elseif lookup_table[object] then
            return lookup_table[object]
        end  -- if
        local new_table = {}
        lookup_table[object] = new_table
        for index, value in pairs(object) do
            new_table[_copy(index)] = _copy(value)
        end  -- for
        return setmetatable(new_table, getmetatable(object))
    end  -- function _copy
    return _copy(object)
end  -- function deepcopy 

EnsembleModel.extend = argcheck{
    {name='self', type='EnsembleModel'},
    {name='n', type='number'},
    call = function(self, n)
        for _, model in ipairs(self.models) do
            model:extend(n)
        end
    end
}

return EnsembleModel
