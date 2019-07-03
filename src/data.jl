using StaticArrays 

function evec(i)
   if i == 1
      return SVector(1.0,0.0,0.0)
   elseif i == 2
      return SVector(0.0,1.0,0.0)
   elseif i == 3
      return SVector(0.0,0.0,1.0)
   end
   @error("`evec`: input must be 1,2 or 3")
end

# maybe move scripts in here eventually
