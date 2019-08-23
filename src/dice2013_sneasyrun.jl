using Mimi
using ExcelReaders

include("helpers.jl")

include("components/grosseconomy_component.jl")
#include("components/emissions_component.jl")
#include("components/co2cycle_component.jl")
#include("components/radiativeforcing_component.jl")
#include("components/climatedynamics_component.jl")
include("components/damages_component.jl")
include("components/neteconomy_component.jl")
#include("components/welfare_component.jl")

function constructdice()
    m = Model()

    setindex(m, :time, collect(2010:5:2300))

    addcomponent(m, grosseconomy)
    #addcomponent(m, emissions)
    #addcomponent(m, co2cycle)
    #addcomponent(m, radiativeforcing)
    #addcomponent(m, climatedynamics)
    addcomponent(m, damages)
    addcomponent(m, neteconomy)
    #addcomponent(m, welfare)


    #GROSS ECONOMY COMPONENT
    setparameter(m, :grosseconomy, :al, getparams("B5:BI5", :all, "DICE2013_Base", 60))
    setparameter(m, :grosseconomy, :l, getparams("B6:BI6", :all, "DICE2013_Base", 60))
    setparameter(m, :grosseconomy, :gama, getparams("B7:B7", :single, "DICE2013_Base", 1))
    setparameter(m, :grosseconomy, :dk, getparams("B8:B8", :single, "DICE2013_Base", 1))
    setparameter(m, :grosseconomy, :k0,  getparams("B9:B9", :single, "DICE2013_Base", 1))

    connectparameter(m, :grosseconomy, :I, :neteconomy, :I)


    #EMISSIONS COMPONENT
    #setparameter(m, :emissions, :sigma, getparams("B12:BI12", :all, "DICE2013_Base", 60))
    #setparameter(m, :emissions, :MIU, getparams("B47:BI47", :all, "DICE2013_Base", 60))
    #setparameter(m, :emissions, :etree, getparams("B13:BI13", :all, "DICE2013_Base", 60))
    #setparameter(m, :emissions, :cca0, getparams("B14:B14", :single, "DICE2013_Base", 1))

    #connectparameter(m, :emissions, :YGROSS, :grosseconomy, :YGROSS)


    #CO2 CYCLE COMPONENT
    #setparameter(m, :co2cycle, :mat0, getparams("B17:B17", :single, "DICE2013_Base", 1))
    #setparameter(m, :co2cycle, :mu0, getparams("B18:B18", :single, "DICE2013_Base", 1))
    #setparameter(m, :co2cycle, :ml0, getparams("B19:B19", :single, "DICE2013_Base", 1))
    #setparameter(m, :co2cycle, :b12, getparams("B20:B20", :single, "DICE2013_Base", 1))
    #setparameter(m, :co2cycle, :b23, getparams("B21:B21", :single, "DICE2013_Base", 1))
    #setparameter(m, :co2cycle, :b11, getparams("B22:B22", :single, "DICE2013_Base", 1))
    #setparameter(m, :co2cycle, :b21, getparams("B23:B23", :single, "DICE2013_Base", 1))
    #setparameter(m, :co2cycle, :b22, getparams("B24:B24", :single, "DICE2013_Base", 1))
    #setparameter(m, :co2cycle, :b32, getparams("B25:B25", :single, "DICE2013_Base", 1))
    #setparameter(m, :co2cycle, :b33, getparams("B26:B26", :single, "DICE2013_Base", 1))

    #connectparameter(m, :co2cycle, :E, :emissions, :E)


    #RADIATIVE FORCING COMPONENT
    #setparameter(m, :radiativeforcing, :forcoth, getparams("B29:BI29", :all, "DICE2013_Base", 60))
    #setparameter(m, :radiativeforcing, :fco22x, getparams("B30:B30", :single, "DICE2013_Base", 1))

    #connectparameter(m, :radiativeforcing, :MAT, :co2cycle, :MAT)


    #CLIMATE DYNAMICS COMPONENT
    #setparameter(m, :climatedynamics, :fco22x, getparams("B30:B30", :single, "DICE2013_Base", 1))
    #setparameter(m, :climatedynamics, :t2xco2, getparams("B33:B33", :single, "DICE2013_Base", 1))
    #setparameter(m, :climatedynamics, :tatm0, getparams("B34:B34", :single, "DICE2013_Base", 1))
    #setparameter(m, :climatedynamics, :tocean0, getparams("B35:B35", :single, "DICE2013_Base", 1))
    #setparameter(m, :climatedynamics, :c1, getparams("B36:B36", :single, "DICE2013_Base", 1))
    #setparameter(m, :climatedynamics, :c3, getparams("B37:B37", :single, "DICE2013_Base", 1))
    #setparameter(m, :climatedynamics, :c4, getparams("B38:B38", :single, "DICE2013_Base", 1))

    #connectparameter(m, :climatedynamics, :FORC, :radiativeforcing, :FORC)


    #DAMAGES COMPONENT
    setparameter(m, :damages, :a1, getparams("B41:B41", :single, "DICE2013_Base", 1))
    setparameter(m, :damages, :a2, getparams("B42:B42", :single, "DICE2013_Base", 1))
    setparameter(m, :damages, :a3, getparams("B43:B43", :single, "DICE2013_Base", 1))

    #connectparameter(m, :damages, :TATM, :climatedynamics, :TATM)
    connectparameter(m, :damages, :YGROSS, :grosseconomy, :YGROSS)


    #NET ECONOMY COMPONENT
    setparameter(m, :neteconomy, :cost1, getparams("B46:BI46", :all, "DICE2013_Base", 60))
    setparameter(m, :neteconomy, :MIU, getparams("B47:BI47", :all, "DICE2013_Base", 60))
    setparameter(m, :neteconomy, :expcost2, getparams("B48:B48", :single, "DICE2013_Base", 1))
    setparameter(m, :neteconomy, :partfract, getparams("B49:BI49", :all, "DICE2013_Base", 60))
    setparameter(m, :neteconomy, :pbacktime, getparams("B50:BI50", :all, "DICE2013_Base", 60))
    setparameter(m, :neteconomy, :S, getparams("B51:BI51", :all, "DICE2013_Base", 60))
    setparameter(m, :neteconomy, :l, getparams("B6:BI6", :all, "DICE2013_Base", 60))

    connectparameter(m, :neteconomy, :YGROSS, :grosseconomy, :YGROSS)
    connectparameter(m, :neteconomy, :DAMFRAC, :damages, :DAMFRAC)


    #WELFARE COMPONENT
    #setparameter(m, :welfare, :l, getparams("B6:BI6", :all, "DICE2013_Base", 60))
    #setparameter(m, :welfare, :elasmu, getparams("B54:BI54", :single, "DICE2013_Base", 1))
    #setparameter(m, :welfare, :rr, getparams("B55:BI55", :all, "DICE2013_Base", 60))
    #setparameter(m, :welfare, :scale1, getparams("B56:B56", :single, "DICE2013_Base", 1))
    #setparameter(m, :welfare, :scale2, getparams("B57:B57", :single, "DICE2013_Base", 1))

    #connectparameter(m, :welfare, :C, :neteconomy, :C)


    #run(m)
    return m

end


function getdice()

    m=constructdice()

    return m
end







