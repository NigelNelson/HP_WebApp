class ToggleGroup extends React.Component {
    render() {
        return (
            <div className="row row-cols-1 w-100 mx-0 p-0"
                 style={{overflowX: 'auto',
                     maxHeight: '370px',
                     outlineStyle: "solid",
                     outlineWidth: 1}}>
                <div className="m-0 p-0 w-100" style={{outlineStyle: "solid", outlineWidth: 1}}>
                    <button type="button"
                            className="btn btn-sm btn-primary row m-0 w-100"
                            onClick={this.props.handleClickAll} enabled>Show All Points
                    </button>
                </div>
                {this.props.toggle_nums.map(num =>
                    <div className="col m-0 p-0 w-100">
                        <PointToggle num={num} handlePointToggle={this.props.handlePointToggle}/>
                    </div>
                )}
            </div>
        )
    }
}