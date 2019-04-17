$("#getButton").click(function(){
user_id=$("#user_id").val();
if(user_id==undefined || user_id=="")
{
alert("please enter value");
return;
}

if(user_id>5 || user_id<1)
{
alert("please enter valid value");
return;
}
window.location.href="/recommend/"+user_id
});

