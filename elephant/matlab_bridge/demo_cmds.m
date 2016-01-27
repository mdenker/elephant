%cmds
isi=call_elephant('isi','elephant.statistics',[1,2,5,76,97]);
load_blackrock;
isi=call_elephant('isi','elephant.statistics',nikos_blk.segments.spiketrains.times');
cv=call_elephant('cv','elephant.statistics',isi');