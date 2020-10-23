macro workunit(title, code, savelist)
    expr = quote
        is_cached = all(i->isfile(i), [$((i.args[3] for i in savelist.args)...)])

        is_cached && println("SKIPPING: ", $title)

        is_cached && return

        println("STARTING: ", $title)

        $code

        println("SAVING: ", $title)

        map(i->save(i[2], i[1]), [$((:($(i.args[2]), $(i.args[3])) for i in savelist.args)...)])

        println("FINISHED: ", $title)
    end

    thunk = esc(:(()->($expr)))
    var = esc(Base.sync_varname)
    spawncall = :($(Distributed.spawn_somewhere)($thunk))
    quote
        local ref = $spawncall
        if $(Expr(:islocal, var))
            put!($var, ref)
        end
        ref
    end
end