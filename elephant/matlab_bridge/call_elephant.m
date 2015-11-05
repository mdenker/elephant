function result=call_elephant(elefun,package,varargin)
    vname=@(x) inputname(1);
    
    cmd=sprintf(['import ' package '\n']);
    
    cmd=sprintf([cmd 'output=' package '.' elefun '(']);
    for i=1:numel(varargin)
        py('set', sprintf(['etmp' num2str(i)]), squeeze(varargin{i}));
        
        if i>1
            cmd=sprintf([cmd ',']);
        end
        
        cmd=sprintf([cmd 'etmp' num2str(i)]);
    end
    
    cmd=sprintf([cmd ')\ndel etmp1']);
    
    disp(sprintf('Executing\n-------------------------------------------'));
    disp(cmd);
    
    py('eval',cmd);
    result=py('get', 'output');
end